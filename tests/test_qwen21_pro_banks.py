"""独立 source bank、身份局部性与人物置换等变性。"""
from dataclasses import replace
import importlib

import torch

from test_qwen21_pro_contracts import api, make_condition


def bank_case():
    torch.manual_seed(12)
    cfg = api().ProAdapterConfig(hidden_dim=32, width=16, heads=4)
    bank = importlib.import_module("src.pose_control.qwen21_pro.banks").PersonBankAttention(cfg)
    return bank, make_condition(), torch.randn(1, 2, 4, 16)


def test_bank_change_is_person_local_and_all_masked_does_not_borrow():
    bank, pack, query = bank_case()
    original = bank(query, pack)
    tokens = pack.source_tokens.clone()
    tokens[:, 1] += 20
    changed = bank(query, replace(pack, source_tokens=tokens))
    assert torch.equal(original[:, 0], changed[:, 0])
    assert not torch.allclose(original[:, 1], changed[:, 1])
    validity = pack.source_token_valid.clone()
    validity[:, 0] = False
    without_id = replace(pack, face_identity_valid=torch.zeros_like(pack.face_identity_valid), source_token_valid=validity)
    null = bank(query, without_id)
    assert torch.equal(null[:, 0], torch.zeros_like(null[:, 0]))
    assert torch.isfinite(null).all()


def test_face_identity_is_gated_by_face_support_and_validity():
    bank, pack, query = bank_case()
    face = torch.zeros_like(pack.face_mask)
    face[:, :, 0] = .01
    pack = replace(pack, face_mask=face)
    original = bank(query, pack)
    changed = bank(query, replace(pack, face_identity=pack.face_identity + 50))
    assert torch.equal(original[:, :, 1:], changed[:, :, 1:])
    assert not torch.allclose(original[:, :, :1], changed[:, :, :1])
    invalid = replace(pack, face_identity_valid=torch.zeros_like(pack.face_identity_valid))
    assert torch.equal(bank(query, invalid), bank(query, replace(invalid, face_identity=pack.face_identity + 50)))


def test_overlapping_geometry_appearance_uses_visible_owner_only():
    torch.manual_seed(4)
    model = api().ProTaskAdapter(api().ProAdapterConfig(hidden_dim=32, width=16, heads=4))
    pack = make_condition()
    with torch.no_grad():
        model.output_heads["0"].weight.normal_()
    # Both masks are amodal/overlapping; ownership belongs wholly to person zero.
    owner = torch.zeros_like(pack.owner)
    owner[:, 0] = 1
    pack = replace(pack, owner=owner)
    x = torch.randn(1, 4, 32)
    kwargs = dict(stage=0, target_mask=torch.ones(4, dtype=torch.bool), timestep=torch.tensor([.2]))
    first = model(x, conditions=pack, **kwargs)
    source = pack.source_tokens.clone()
    source[:, 1] += 30
    assert torch.equal(first, model(x, conditions=replace(pack, source_tokens=source), **kwargs))


def test_full_person_relabel_is_equivariant_including_contact_endpoints():
    torch.manual_seed(9)
    model = api().ProTaskAdapter(api().ProAdapterConfig(hidden_dim=32, width=16, heads=4))
    pack = make_condition()
    relation = api().ContactRelations(torch.tensor([[0]]), torch.tensor([[3]]), torch.tensor([[1]]), torch.tensor([[7]]), torch.randn(1, 1, 6), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool))
    pack = replace(pack, relations=relation)
    perm = torch.tensor([1, 0])
    fields = {name: getattr(pack, name)[:, perm] for name in ("person_mask", "owner", "face_mask", "hand_mask", "person_valid", "source_tokens", "source_token_valid", "face_identity", "face_identity_valid")}
    shuffled = replace(pack, person_ids=tuple(reversed(pack.person_ids)), spatial={k: v[:, perm] for k, v in pack.spatial.items()}, spatial_valid={k: v[:, perm] for k, v in pack.spatial_valid.items()}, relations=replace(relation, src_person=1-relation.src_person, dst_person=1-relation.dst_person), **fields)
    with torch.no_grad():
        model.output_heads["0"].weight.normal_()
    x = torch.randn(1, 4, 32)
    kwargs = dict(stage=0, target_mask=torch.ones(4, dtype=torch.bool), timestep=torch.tensor([.2]))
    torch.testing.assert_close(model(x, conditions=pack, **kwargs), model(x, conditions=shuffled, **kwargs), rtol=1e-6, atol=1e-6)


def test_contact_encoder_preserves_endpoint_pairing():
    pack = make_condition()
    cfg = api().ProAdapterConfig(hidden_dim=32, width=16, heads=4)
    encoder = importlib.import_module("src.pose_control.qwen21_pro.encoders").ContactEncoder(cfg)
    relation = api().ContactRelations(torch.tensor([[0]]), torch.tensor([[3]]), torch.tensor([[1]]), torch.tensor([[7]]), torch.zeros(1, 1, 6), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool))
    first = encoder(replace(pack, relations=relation))
    second = encoder(replace(pack, relations=replace(relation, src_part=relation.dst_part, dst_part=relation.src_part)))
    assert not torch.allclose(first, second)
    assert torch.equal(encoder(pack), torch.zeros(1, 2, 16))


def test_bank_swap_follows_source_binding_and_empty_source_has_no_nan():
    bank, pack, query = bank_case()
    query = query[:, :1].expand(-1, 2, -1, -1)
    pack = replace(pack, face_identity_valid=torch.zeros_like(pack.face_identity_valid))
    original = bank(query, pack)
    swapped = bank(query, replace(pack, source_tokens=pack.source_tokens.flip(1)))
    torch.testing.assert_close(original.flip(1), swapped, rtol=1e-6, atol=1e-6)
    empty = replace(pack, source_tokens=pack.source_tokens[:, :, :0], source_token_valid=pack.source_token_valid[:, :, :0])
    assert torch.equal(bank(query, empty), torch.zeros_like(query))

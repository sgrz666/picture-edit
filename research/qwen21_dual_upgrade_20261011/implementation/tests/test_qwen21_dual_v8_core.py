from dataclasses import fields, replace

import pytest
import torch

from src.pose_control.qwen21_full.dual import DualV8Adapter
from src.pose_control.qwen21_full.dual.conditions import ControlStrengths
from src.pose_control.qwen21_full.dual.checkpoint import load_checkpoint, save_checkpoint, warm_start_from_body
from src.pose_control.qwen21_full.dual.routing import compute_routing
from test_qwen21_dual_v8_conditions import make_input

torch.set_num_threads(1)


def adapter():
    torch.manual_seed(73)
    return DualV8Adapter(width=32, hidden_size=64, rank=8, layer_indices=(3, 8))


def activate_heads(model):
    with torch.no_grad():
        for heads in model.output_heads.values():
            for head in heads.values():
                head[-1].weight.normal_(std=.02)


def test_part_nodes_encode_anatomy_orientation_joint_and_source_shape():
    model = adapter().eval()
    geometry = make_input().geometry
    assert model.reasoner.part_embedding.num_embeddings == 14
    assert model.reasoner.part_descriptor[0].in_features == 26
    assert model.reasoner.mutual[0].ffn[1].out_features == 4*model.width
    observed = []
    hook = model.reasoner.part_descriptor.register_forward_pre_hook(lambda _,args: observed.append(args[0].detach()))
    state = model.prepare(make_input())
    hook.remove()
    torch.testing.assert_close(observed[0][...,-10:],geometry.global32[...,:10,None].transpose(-1,-2).expand(-1,-1,14,-1))
    for name in ('part_orientations','part_joints'):
        value = make_input()
        field = getattr(value.geometry,name).clone()
        if name == 'part_orientations':
            field[0,0,1] = torch.tensor([1.,0.,0.])
        else:
            field[0,0,1,0] += .5
        changed = replace(value,geometry=replace(value.geometry,**{name:field},allow_test_resolution=True))
        assert not torch.allclose(state.nodes,model.prepare(changed).nodes)


def test_appearance_modality_embedding_has_distinct_banks_and_invalid_exact_zero():
    model = adapter().eval()
    encoder = model.appearance_encoder
    assert encoder.modality_embedding.num_embeddings == 2
    value = make_input(with_b=False,contact=False)
    before,_ = encoder(value.source)
    with torch.no_grad():
        encoder.modality_embedding.weight[0].add_(2)
    after,valid = encoder(value.source)
    difference=after[:,0,:8]-before[:,0,:8]
    torch.testing.assert_close(difference,valid[:,0,:8,None].float().expand_as(difference)*2)
    torch.testing.assert_close(before[:,0,8:],after[:,0,8:],rtol=0,atol=0)
    assert not valid[:,1].any() and after[:,1].count_nonzero()==0


@pytest.mark.parametrize('key',['geometry_encoder_version','appearance_encoder_version','partition_sha256','coordinate_convention','rotation6d_convention','loading_report'])
def test_checkpoint_requires_exact_encoder_provenance(tmp_path,key):
    path=tmp_path/'strict.pt'
    model=adapter()
    saved=save_checkpoint(path,model,{'model_id':'Qwen/Qwen-Image-2.1'},step=7)
    assert key in saved['metadata']
    broken=dict(saved)
    broken['metadata']=dict(saved['metadata'])
    broken['metadata'].pop(key)
    torch.save(broken,path)
    with pytest.raises(ValueError,match='provenance'):
        load_checkpoint(path)


def test_zero_init_budget_has_finite_gradient():
    model = adapter()
    state = model.prepare(make_input())
    delta = model.residual(3,torch.randn(1,4,64),state,.8,.8)
    (delta.square().mean()+model.budget_loss(.05)).backward()
    for name,parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(),name


def test_cuda_autocast_prepare_and_flow_backward_keep_master_parameters_float32():
    if not torch.cuda.is_available():pytest.skip('requires production CUDA mixed precision')
    from src.pose_control.qwen21_full.dual.runtime import prepared_for_qwen
    model=adapter().cuda()
    value=make_input().to('cuda',torch.float32)
    state=prepared_for_qwen(model,value)
    assert state.dense.dtype==torch.bfloat16 and state.contacts.dtype==torch.bfloat16
    with torch.autocast('cuda',dtype=torch.bfloat16):
        delta=model.residual(3,torch.randn(1,4,64,device='cuda',dtype=torch.bfloat16),state,.65,.65)
        loss=(delta.float()-1).square().mean()+model.budget_loss()
    loss.backward()
    assert model.output_heads['3']['geometry'][-1].weight.grad.abs().sum()>0
    for p in model.parameters():
        assert p.dtype==torch.float32
        if p.grad is not None:assert torch.isfinite(p.grad).all()


def test_routing_weights_preserve_float32_conservation_under_bfloat16_transfer():
    state=adapter().prepare(make_input())
    torch.manual_seed(44)
    parts=torch.rand(1,3,4,1)
    parts=parts/parts.sum(1,keepdim=True)
    state=replace(state,weights_geometry=parts[:,:2],weights_contact=parts[:,2])
    converted=state.to('cpu',torch.bfloat16)
    assert converted.weights_geometry.dtype==torch.float32
    assert converted.weights_contact.dtype==torch.float32
    assert converted.weights_appearance.dtype==torch.float32
    assert torch.equal(converted.weights_geometry,state.weights_geometry)


def test_empty_appearance_bank_bypasses_biased_ffn_and_head():
    model = adapter()
    activate_heads(model)
    state = model.prepare(make_input())
    state = replace(state,appearance=torch.zeros_like(state.appearance),
                    appearance_valid=torch.zeros_like(state.appearance_valid))
    output = model.residual(3,torch.randn(1,4,64),state,.8,.8,ControlStrengths(0,1,0))
    assert output.count_nonzero()==0


def test_geometry_dilation_competes_by_depth_in_expanded_overlap():
    value = make_input(contact=False)
    g = value.geometry
    masks = torch.zeros_like(g.amodal_mask)
    masks[:,0,:,:,3:8]=1
    masks[:,1,:,:,20:25]=1
    g = replace(g,amodal_mask=masks,visible_owner=masks,
        part=masks*torch.nn.functional.one_hot(torch.tensor(1),14)[None,None,:,None,None],
        depth=masks*torch.tensor([2.,3.])[None,:,None,None,None],allow_test_resolution=True)
    geo,app,_ = compute_routing(g,(32,32))
    # x=14 is in both eight-pixel dilations, but belongs to the nearer surface.
    assert geo[0,0,16*32+14,0]==1 and geo[0,1,16*32+14,0]==0
    assert app[0,:,16*32+14].count_nonzero()==0


def test_contact_graph_has_eight_head_edge_attention_and_confidence_weights():
    from src.pose_control.qwen21_full.dual.reasoner import ContactGraphLayer
    layer = ContactGraphLayer(32)
    assert getattr(layer,'heads',None)==8
    nodes,edges = torch.randn(1,28,32),torch.randn(1,1,32)
    valid = torch.ones(1,28,dtype=torch.bool)
    endpoints = torch.tensor([[[1,15]]])
    edge_valid = torch.ones(1,1,dtype=torch.bool)
    high = layer(nodes,valid,endpoints,edges,edge_valid,torch.ones(1,1))[0]
    low = layer(nodes,valid,endpoints,edges,edge_valid,torch.full((1,1),.01))[0]
    assert not torch.allclose(high,low)


def test_residual_diagnostics_measure_only_active_roi():
    model = adapter()
    activate_heads(model)
    state = model.prepare(make_input(contact=False))
    weights = torch.zeros_like(state.weights_geometry)
    weights[:,0,0]=1
    state = replace(state,weights_geometry=weights)
    hidden = torch.randn(1,4,64)
    output = model.residual(3,hidden,state,.8,.8,ControlStrengths(1,0,0))
    diag = model.last_diagnostics[3]['geometry']
    expected = float(output[:,0].square().mean().sqrt())
    assert diag['rms']==pytest.approx(expected)
    assert diag['ratio']==pytest.approx(expected/float(hidden[:,0].square().mean().sqrt()))


def permute(value):
    source_values = {f.name: getattr(value.source, f.name) for f in fields(value.source)}
    for name in ("masks", "person_latents", "latent_masks", "dino_patches", "dino_valid"):
        source_values[name] = source_values[name].flip(1)
    source_values["person_uids"] = tuple(tuple(reversed(row)) for row in value.source.person_uids)
    geometry_values = {f.name: getattr(value.geometry, f.name) for f in fields(value.geometry)}
    for name in ("normal", "depth", "part", "pose", "amodal_mask", "visible_owner", "global32", "relative14", "part_positions", "part_orientations", "part_joints", "part_valid", "part_confidence", "person_valid"):
        geometry_values[name] = geometry_values[name].flip(1)
    geometry_values["person_uids"] = source_values["person_uids"]
    geometry_values["contact_endpoints"] = (geometry_values["contact_endpoints"] + 14) % 28
    return replace(value,
        source=type(value.source)(**source_values, allow_test_resolution=True),
        geometry=type(value.geometry)(**geometry_values, allow_test_resolution=True))


def test_prepare_has_per_person_banks_and_geometry_cannot_see_appearance():
    model = adapter().eval()
    value = make_input()
    state = model.prepare(value)
    assert state.dense.shape == (1, 2, 4, 32)
    assert state.nodes.shape == (1, 2, 14, 32)
    assert state.appearance.shape == (1, 2, 16, 32)
    assert state.contacts.shape == (1, 1, 32)
    changed = replace(value, source=replace(value.source, person_latents=value.source.person_latents+4, dino_patches=value.source.dino_patches+2, allow_test_resolution=True))
    other = model.prepare(changed)
    assert torch.equal(state.dense, other.dense)
    assert torch.equal(state.nodes, other.nodes)
    assert torch.equal(state.contacts, other.contacts)
    assert not torch.allclose(state.appearance, other.appearance)
    assert state.to("cpu", torch.float64).node_valid.dtype == torch.bool


def test_true_zero_init_then_final_head_and_upstream_gradient():
    model = adapter()
    value = make_input()
    hidden = torch.randn(1, 4, 64)
    optim = torch.optim.SGD(model.parameters(), lr=.05)
    state = model.prepare(value)
    delta = model.residual(3, hidden, state, timestep=800., sigma=.8)
    assert torch.count_nonzero(delta) == 0
    goal = torch.randn_like(hidden)
    (delta-goal).square().mean().backward()
    final = model.output_heads["3"]["geometry"][-1]
    assert final.weight.grad.abs().sum() > 0
    assert model.query_projection.weight.grad.abs().sum() == 0
    optim.step()
    optim.zero_grad(set_to_none=True)
    state = model.prepare(value)
    delta = model.residual(3, hidden, state, timestep=800., sigma=.8)
    (delta-goal).square().mean().backward()
    assert model.query_projection.weight.grad.abs().sum() > 0
    assert model.geometry_encoder.depth_visibility[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("with_b,contact", [(True, True), (True, False), (False, False)])
def test_slot_permutation_equivariance_including_contact_endpoints(with_b, contact):
    model = adapter().eval()
    activate_heads(model)
    value = make_input(with_b=with_b, contact=contact)
    state, swapped = model.prepare(value), model.prepare(permute(value))
    for name in ("dense", "nodes", "appearance", "weights_geometry", "weights_appearance", "person_valid"):
        torch.testing.assert_close(getattr(state, name).flip(1), getattr(swapped, name), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state.contacts, swapped.contacts, atol=2e-5, rtol=2e-5)
    hidden = torch.randn(1, 4, 64)
    torch.testing.assert_close(model.residual(3, hidden, state, 400., .4), model.residual(3, hidden, swapped, 400., .4), atol=2e-5, rtol=2e-5)


def test_depth_routing_uses_visible_owner_and_symmetric_nearest_depth():
    value = make_input(contact=False)
    geometry = value.geometry
    mask = torch.ones_like(geometry.amodal_mask)
    owner = torch.zeros_like(mask)
    owner[:, 0] = 1
    near_a = replace(geometry, amodal_mask=mask, visible_owner=owner, depth=mask*torch.tensor([2.,3.])[None,:,None,None,None], allow_test_resolution=True)
    geo, app, contact = compute_routing(near_a, (2, 2))
    assert torch.count_nonzero(geo[:, 1]) == 0
    assert torch.count_nonzero(app[:, 1]) == 0
    assert torch.count_nonzero(contact) == 0
    assert torch.all(geo.sum(1)+contact <= 1.000001)
    tied = replace(near_a, depth=mask*2, allow_test_resolution=True)
    geo, _, _ = compute_routing(tied, (2, 2))
    torch.testing.assert_close(geo[:, 0], geo[:, 1])


def test_empty_contact_and_missing_b_exact_zero_and_strengths_independent():
    model = adapter().eval()
    activate_heads(model)
    hidden = torch.randn(1, 4, 64)
    for with_b in (True, False):
        state = model.prepare(make_input(with_b=with_b, contact=False))
        assert state.contacts.shape[1] == 0
        assert torch.count_nonzero(state.weights_contact) == 0
        delta = model.residual(3, hidden, state, 400., .4, ControlStrengths(0, 0, 1))
        assert torch.count_nonzero(delta) == 0
        if not with_b:
            for name in ("dense", "nodes", "appearance", "weights_geometry", "weights_appearance"):
                assert torch.count_nonzero(getattr(state, name)[:, 1]) == 0
            assert not state.appearance_valid[:, 1].any()
    state = model.prepare(make_input())
    branches = [model.residual(3, hidden, state, 400., .4, strengths) for strengths in (ControlStrengths(1,0,0),ControlStrengths(0,1,0),ControlStrengths(0,0,1))]
    total = model.residual(3, hidden, state, 400., .4)
    torch.testing.assert_close(total, sum(branches))
    assert torch.count_nonzero(model.residual(3, hidden, state, 400., .4, ControlStrengths(0,0,0))) == 0


def test_local_rms_cap_and_diagnostics_budget_loss():
    model = adapter().eval()
    activate_heads(model)
    state = model.prepare(make_input())
    hidden = torch.randn(1, 4, 64)
    delta = model.residual(3, hidden, state, torch.tensor([400.]), torch.tensor([.4]), ControlStrengths(max_ratio=.01))
    assert torch.all(delta.float().square().mean(-1).sqrt() <= hidden.float().square().mean(-1).sqrt()*.01001)
    diagnostics = model.last_diagnostics[3]
    for branch in ("geometry", "appearance", "contact", "total"):
        assert {"rms", "max", "count", "cumulative"} <= diagnostics[branch].keys()
    loss = model.budget_loss()
    assert loss.requires_grad and torch.isfinite(loss)


def test_checkpoint_strict_roundtrip_and_warm_start_scopes(tmp_path):
    model = adapter()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    spatial_key = next(name for name in before if name.startswith("geometry_encoder.spatial_encoder.normal_stem.") and name.endswith("weight"))
    person_key = "reasoner.person_reasoner.input_projection.weight"
    payload = {"adapter_state_dict": {
        spatial_key.replace("geometry_encoder.spatial_encoder.", "condition_injector.spatial_encoder."): torch.full_like(before[spatial_key], .25),
        person_key: torch.full_like(before[person_key], .3),
        "reasoner.role_embedding.weight": torch.full((2,32), .4),
        "condition_injector.global_encoder.mlp.0.weight": torch.ones(256,26),
        "control_interface.control_core.head.weight": torch.ones(2,2),
    }, "step":19500}
    report = warm_start_from_body(model, payload)
    assert len(report["matched"]) == 3
    assert len(report["skipped"]) == 2
    assert torch.all(model.state_dict()[spatial_key] == .25)
    assert torch.all(model.state_dict()[person_key] == .3)
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before if k not in {spatial_key,person_key,"reasoner.shared_role_offset"})
    path = tmp_path / "dual.pt"
    save_checkpoint(path, model, {"model_id":"Qwen/Qwen-Image-Edit-2511", "source_checkpoint_step":19500}, step=2)
    loaded, metadata, payload = load_checkpoint(path)
    assert metadata["architecture_version"] == "dual_v8"
    assert metadata["layer_indices"] == [3,8]
    assert payload["step"] == 2
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key])
    broken = dict(payload)
    broken["metadata"] = dict(metadata, architecture_version="v6.6")
    torch.save(broken,path)
    with pytest.raises(ValueError, match="dual_v8"):
        load_checkpoint(path)

import torch
from scripts.data.initialize_qwen21_fixedpair_v5 import adapt_face72_order


def test_canonical_face_order_preserves_index_embeddings_and_spatial_convolution():
    torch.manual_seed(1)
    prefix='legacy.face_preparer.'
    emb=torch.randn(72,32);w=torch.randn(16,73,3,3)
    source={prefix+'geometry_tokenizer.landmark_index.weight':emb,
        prefix+'spatial_encoder.stem.weight':w}
    state,report=adapt_face72_order(source)
    order=torch.tensor([*range(4),*range(55,72),*range(4,55)])
    assert torch.equal(state[prefix+'geometry_tokenizer.landmark_index.weight'],emb[order])
    x=torch.randn(1,73,9,9);new_x=torch.cat((x[:,order],x[:,72:]),1)
    torch.testing.assert_close(torch.nn.functional.conv2d(x,w),
        torch.nn.functional.conv2d(new_x,state[prefix+'spatial_encoder.stem.weight']),atol=3e-5,rtol=1e-4)
    assert torch.equal(source[prefix+'spatial_encoder.stem.weight'],w)
    assert len(report['adapted_tensors'])==2

import json
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.letterbox import Letterbox, valid_body_velocity_loss
from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache


def test_shared_image_point_camera_coordinates():
    transform = Letterbox.from_size((604,1080),512)
    assert transform.rect == (113,0,399,512)
    image = Image.new('L',(604,1080))
    arr=np.zeros((1080,604),dtype=np.uint8)
    arr[450:630,250:354]=255
    image=Image.fromarray(arr)
    warped=np.asarray(transform.image(image,nearest=True))
    center=transform.points(torch.tensor([[.5,.5,1.]]))[0]
    assert warped[round(float(center[1])*512),round(float(center[0])*512)] == 255
    assert not warped[:,:113].any()
    # Camera projection and mapped normalized keypoint are the same operation.
    vector=torch.arange(26,dtype=torch.float32)/10
    vector[19:23]=torch.tensor([1.2,1.3,.5,.5])
    camera=transform.global_camera(vector)
    xy=torch.tensor([.1,-.1])
    projected=vector[19:21]*xy+vector[21:23]
    expected=transform.points(projected)
    actual=camera[19:21]*xy+camera[21:23]
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(camera[:19],vector[:19],rtol=0,atol=0)
    torch.testing.assert_close(camera[23:],vector[23:],rtol=0,atol=0)


@pytest.mark.parametrize('size',[(320,640),(640,320),(512,512),(111,333)])
def test_padding_records_pixels_and_preserves_labels(size):
    transform=Letterbox.from_size(size,512)
    labels=Image.fromarray(np.full((32,32),7,dtype=np.uint8))
    result=np.asarray(transform.image(labels,nearest=True))
    assert set(np.unique(result)) <= {0,7}
    assert int(transform.valid_mask().sum()) == transform.resized_width*transform.resized_height
    rgb=transform.image(Image.new('RGB',size,(10,20,30)),fill=(128,128,128))
    assert rgb.size==(512,512)
    if size==(512,512):
        assert rgb.tobytes()==Image.new('RGB',size,(10,20,30)).tobytes()


def test_padding_cannot_improve_body_loss_or_receive_gradient():
    squared=torch.tensor([[[1.,1.],[1000.,1000.]]],requires_grad=True)
    human=torch.zeros(1,2,1)
    valid=torch.tensor([[[1.],[0.]]])
    loss=valid_body_velocity_loss(squared,human,valid)
    assert loss.item()==1
    loss.backward()
    assert not squared.grad[:,1].any()
    legacy=valid_body_velocity_loss(squared.detach(),human)
    assert legacy.item()==500.5


def test_cache_namespace_isolates_letterbox_even_for_square_rgb(tmp_path):
    image=Image.new('RGB',(512,512),(128,128,128))
    legacy=Qwen21IPERCache(tmp_path/'legacy')
    new=Qwen21IPERCache(tmp_path/'new',preprocessing='letterbox_v1')
    assert legacy._path(image,'source').name != new._path(image,'source').name
    assert json.loads((tmp_path/'new/cache_policy.json').read_text())['preprocessing']=='letterbox_v1'
    with pytest.raises(ValueError,match='policy differs'):
        Qwen21IPERCache(tmp_path/'new',preprocessing='another_policy')

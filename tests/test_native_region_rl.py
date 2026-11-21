"""Flow-SDE replay and masked group advantages for real generated-image RL."""
import pytest
import torch


def test_flow_sde_replay_matches_quantized_rollout_and_has_gradient():
    from src.pose_control.qwen21_full.region_rl import flow_sde_transition
    x=torch.randn(1,8,4).to(torch.bfloat16);v=torch.randn(1,8,4,requires_grad=True)
    y,old,_,_=flow_sde_transition(x,v.detach(),.7,.6,noise_level=.7,generator=torch.Generator().manual_seed(9))
    assert y.dtype==x.dtype
    _,new,_,_=flow_sde_transition(x,v,.7,.6,noise_level=.7,next_latent=y)
    assert torch.equal(old,new)
    (-new.mean()).backward();assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0


def test_group_advantages_ignore_inapplicable_reward_components():
    from src.pose_control.qwen21_full.region_rl import group_advantages
    rewards={'body':torch.tensor([.1,.3,.5,.7]),'identity':torch.tensor([0.,0.,0.,0.])}
    a=group_advantages(rewards,{'body':1.,'identity':.2},{'body':True,'identity':False})
    torch.testing.assert_close(a.mean(),torch.tensor(0.),atol=1e-6,rtol=0)
    assert a[-1]>a[0] and torch.isfinite(a).all()


def test_constant_reward_does_not_create_artificial_advantages():
    from src.pose_control.qwen21_full.region_rl import group_advantages
    a=group_advantages({'body':torch.ones(4)},{'body':1.},{'body':True})
    assert torch.equal(a,torch.zeros(4))


def test_ppo_clip_and_reference_penalty_are_finite():
    from src.pose_control.qwen21_full.region_rl import grpo_loss
    logp=torch.tensor([-.4,-.2],requires_grad=True)
    loss,parts=grpo_loss(logp,logp.detach(),torch.tensor([-1.,1.]),torch.ones(2),torch.zeros(2),clip=.01,beta=.001)
    loss.backward();assert torch.isfinite(loss) and logp.grad.abs().sum()>0
    assert parts['ratio_min']==parts['ratio_max']==1.


@pytest.mark.parametrize('sigma,next_sigma',[(0.,0.),(.5,.6),(1.,.9)])
def test_invalid_stochastic_transition_is_rejected(sigma,next_sigma):
    from src.pose_control.qwen21_full.region_rl import flow_sde_transition
    with pytest.raises(ValueError):flow_sde_transition(torch.ones(1,2,1),torch.zeros(1,2,1),sigma,next_sigma)

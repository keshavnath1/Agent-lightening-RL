from agent_rewards.policy_reward import get_reward_fn, ruler_relative_reward


def test_reward_wrapper_imports_reward_functions():
    assert callable(get_reward_fn)
    assert callable(ruler_relative_reward)

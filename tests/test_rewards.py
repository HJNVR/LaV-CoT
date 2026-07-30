from lavcot.training.rewards import answer_reward, count_reward, format_reward


def test_rewards():
    outputs = ["<think>English \\\\obj{2}</think><answer>blue</answer>"]
    assert format_reward(outputs) == [1.0]
    assert answer_reward(outputs, ["Blue"]) == [1.0]
    assert count_reward(outputs, [2]) == [1.0]

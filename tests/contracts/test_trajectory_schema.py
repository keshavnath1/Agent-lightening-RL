from agent_contracts import RewardBreakdown, ToolCallRecord, Trajectory, TrajectoryGroup, validate_reward_breakdown


def test_trajectory_contract_round_trip():
    reward = validate_reward_breakdown({"final_reward": 0.8, "valid_json": 1.0, "workflow_score": 0.7, "tool_score": 0.6})
    traj = Trajectory(
        task_id="task-1",
        policy="baseline",
        prompt="p",
        completion="c",
        tool_calls=[ToolCallRecord(tool_name="postgres_get_dataset_schema", status="ok")],
        reward=reward,
    )
    restored = Trajectory.from_dict(traj.to_dict())
    assert restored.task_id == "task-1"
    assert restored.tool_calls[0].tool_name == "postgres_get_dataset_schema"
    assert restored.reward.final_reward == 0.8


def test_trajectory_group_contract():
    group = TrajectoryGroup(task_id="task-1", trajectories=[Trajectory(task_id="task-1", policy="p")])
    assert group.to_dict()["trajectories"][0]["policy"] == "p"

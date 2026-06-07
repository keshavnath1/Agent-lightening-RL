from agent_contracts import RULERScore, RULERScoredGroup, Trajectory, TrajectoryGroup
from ruler_scorer.vllm_judge import score_group_with_vllm_judge


def test_ruler_contract_and_vllm_judge_wrapper_import():
    group = TrajectoryGroup(task_id="task-1", trajectories=[Trajectory(task_id="task-1", policy="p")])
    scored = RULERScoredGroup(
        task_id="task-1",
        group=group,
        scores=[RULERScore(trajectory_id="0", rank=1, ruler_relative_score=1.0)],
        judge_mode="vllm_judge",
    )
    assert scored.to_dict()["scores"][0]["ruler_relative_score"] == 1.0
    assert callable(score_group_with_vllm_judge)

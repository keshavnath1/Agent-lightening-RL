from __future__ import annotations

import argparse
import asyncio
import json
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from src.training.art_official_ml_rollout import MLWorkflowScenario, rollout


def get_ruler_score_group() -> Callable[..., Any]:
    candidates = [
        ("art.rewards", "ruler_score_group"),
        ("art.ruler", "ruler_score_group"),
        ("art.utils", "ruler_score_group"),
    ]
    errors: list[str] = []
    for mod_name, attr in candidates:
        try:
            mod = import_module(mod_name)
            fn = getattr(mod, attr)
            return fn
        except Exception as exc:
            errors.append(f"{mod_name}.{attr}: {exc.__class__.__name__}: {exc}")
    raise RuntimeError(
        "Official RULER function not found. Install/update openpipe-art[backend,langgraph]>=0.4.9. "
        "No heuristic RULER fallback is available. " + " | ".join(errors)
    )


def _load_art() -> Any:
    try:
        import art  # type: ignore
        from art.langgraph import wrap_rollout  # noqa: F401
        return art
    except Exception as exc:
        raise RuntimeError("Official ART/RULER training requires openpipe-art[backend,langgraph]>=0.4.9.") from exc


def _load_scenarios(path: str | Path, limit: int | None = None) -> list[MLWorkflowScenario]:
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenarios: list[MLWorkflowScenario] = []
    for idx, row in enumerate(rows[:limit] if limit else rows):
        scenarios.append(MLWorkflowScenario(
            task_id=str(row.get("task_id") or row.get("id") or f"scenario_{idx:04d}"),
            task_payload=row,
            expected_artifacts=list(row.get("expected_artifacts") or []),
            difficulty=row.get("difficulty"),
        ))
    return scenarios


async def train_with_official_art_ruler(
    scenarios: list[MLWorkflowScenario],
    model_name: str,
    output_dir: str,
    judge_model: str = "openai/o4-mini",
    rollouts_per_group: int = 4,
    groups_per_step: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    max_steps: int = 5,
) -> dict[str, Any]:
    art = _load_art()
    from art.langgraph import wrap_rollout
    from art.utils import iterate_dataset
    ruler_score_group = get_ruler_score_group()
    ruler_source = "official"

    model = art.Model(name=model_name)
    backend = art.LocalBackend()
    await model.register(backend)
    step_count = 0
    training_iterator = iterate_dataset(scenarios, groups_per_step=groups_per_step, num_epochs=num_epochs, initial_step=await model.get_step())
    for batch in training_iterator:
        groups = []
        for scenario in batch.items:
            if hasattr(scenario, "copy"):
                scenario_for_step = scenario.copy(update={"step": batch.step})
            else:
                scenario_for_step = MLWorkflowScenario(**{**scenario.model_dump(), "step": batch.step})
            groups.append(art.TrajectoryGroup([
                wrap_rollout(model, rollout)(model, scenario_for_step)
                for _ in range(rollouts_per_group)
            ]))
        finished_groups = await art.gather_trajectory_groups(groups, pbar_desc="gather", max_exceptions=rollouts_per_group * len(batch.items))
        judged_groups = []
        for group in finished_groups:
            judged_groups.append(await ruler_score_group(group, judge_model))
        result = await backend.train(model, judged_groups, learning_rate=learning_rate, logprob_calculation_chunk_size=8)
        await model.log(judged_groups, metrics=result.metrics, step=result.step, split="train")
        step_count = int(batch.step)
        if batch.step >= max_steps:
            break

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trainer": "official_art_ruler",
        "model_name": model_name,
        "judge_model": judge_model,
        "rollouts_per_group": rollouts_per_group,
        "groups_per_step": groups_per_step,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "steps_completed": step_count,
        "ruler_source": ruler_source,
        "status": "success",
        "output_dir": str(out),
        "adapter_status": "trained",
    }
    (out / "art_training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = Path("reports/art_ruler_training_summary.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Official ART + RULER Training Summary\n\n" + json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_from_dataset(dataset: str, output_dir: str, model_name: str, judge_model: str, rollouts_per_group: int, groups_per_step: int, max_steps: int, limit: int | None = None) -> dict[str, Any]:
    scenarios = _load_scenarios(dataset, limit=limit)
    return asyncio.run(train_with_official_art_ruler(
        scenarios=scenarios,
        model_name=model_name,
        output_dir=output_dir,
        judge_model=judge_model,
        rollouts_per_group=rollouts_per_group,
        groups_per_step=groups_per_step,
        max_steps=max_steps,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official ART + RULER training path when openpipe-art is installed. Fails closed when official RULER is unavailable.")
    parser.add_argument("--dataset", default="data/synthetic/tasks.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/trackb_official_art_ruler")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--judge-model", default="openai/o4-mini")
    parser.add_argument("--rollouts-per-group", type=int, default=4)
    parser.add_argument("--groups-per-step", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    metadata = run_from_dataset(args.dataset, args.output_dir, args.model_name, args.judge_model, args.rollouts_per_group, args.groups_per_step, args.max_steps, args.limit)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

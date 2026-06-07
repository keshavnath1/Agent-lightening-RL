from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
failures: list[str] = []
def check(name: str, ok: bool, detail: str = '') -> None:
    print(('  [PASS] ' if ok else '  [FAIL] ') + name + (f' — {detail}' if detail and not ok else ''))
    if not ok: failures.append(name)
print('PDF alignment validation')
for rel in ['src/telemetry/art_compat.py','src/rewards/apply_ruler_scores.py','src/training/checkpoint_forking.py','src/evaluation/policy_drift.py','src/tools/mcp_discovery.py','src/training/generate_tool_scenarios.py','src/evaluation/tool_use_metrics.py','src/evaluation/flywheel_report.py','src/training/train_policy_qlora_grpo.py']:
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(ROOT / rel)], capture_output=True, text=True)
    check(f'compile {rel}', r.returncode == 0, r.stderr.strip())
try:
    from src.rewards.policy_reward import REWARD_MODES, get_reward_fn
    check('reward mode ruler_relative is registered', 'ruler_relative' in REWARD_MODES)
    got = get_reward_fn('ruler_relative')(['p'], ['{}'], final_hybrid_reward=[0.88])[0]
    check('ruler_relative consumes final_hybrid_reward', abs(got - 0.88) < 0.001, str(got))
except Exception as exc:
    check('reward registry import/call', False, str(exc))
for mod in ['src.rewards.apply_ruler_scores','src.evaluation.policy_drift','src.training.generate_tool_scenarios','src.evaluation.tool_use_metrics','src.evaluation.flywheel_report']:
    r = subprocess.run([sys.executable, '-m', mod, '--help'], cwd=str(ROOT), capture_output=True, text=True)
    check(f'{mod} --help works', r.returncode == 0, r.stderr.strip())
for rel in ['src/ui/pages/overview.py','src/ui/pages/results.py','src/ui/pages/debug.py','src/ui/components/trainer_cards.py']:
    text = (ROOT / rel).read_text(encoding='utf-8')
    check(f'dashboard mentions alignment: {rel}', ('ART' in text or 'RULER' in text), '')
for sh in list((ROOT/'scripts/cpu').glob('*.sh')) + list((ROOT/'scripts/gpu').glob('*.sh')):
    r = subprocess.run(['bash', '-n', str(sh)], capture_output=True, text=True)
    check(f'bash -n {sh.name}', r.returncode == 0, r.stderr.strip())
if failures:
    print(f'PDF alignment validation FAILED: {len(failures)} issue(s).')
    sys.exit(1)
print('PDF alignment validation passed.')

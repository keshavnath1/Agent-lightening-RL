from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
failures: list[str] = []
def check(name: str, ok: bool, detail: str = '') -> None:
    print(('  [PASS] ' if ok else '  [FAIL] ') + name + (f' — {detail}' if detail and not ok else ''))
    if not ok: failures.append(name)
print('Official ART/RULER validation')
for rel in ['src/training/art_availability.py','src/training/art_official_ml_rollout.py','src/training/art_ruler_training.py','src/training/train_policy_qlora_grpo.py']:
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(ROOT / rel)], capture_output=True, text=True)
    check(f'compile {rel}', r.returncode == 0, r.stderr.strip())
try:
    from src.training.art_availability import check_art_available
    status = check_art_available()
    print('  ART availability:', status)
except Exception as exc:
    status = {'art': False, 'art_langgraph': False, 'ruler': False, 'errors': [str(exc)]}
    check('check_art_available import', False, str(exc))
r = subprocess.run([sys.executable, '-m', 'src.training.train_policy_qlora_grpo', '--help'], cwd=str(ROOT), capture_output=True, text=True)
help_text = r.stdout + r.stderr
for token in ['official_art_ruler','--judge-model','--rollouts-per-group','--groups-per-step','--max-steps','No heuristic RULER fallback is available']:
    check(f'help contains {token}', token in help_text)
rollout_src = (ROOT/'src/training/art_official_ml_rollout.py').read_text(encoding='utf-8')
training_src = (ROOT/'src/training/art_ruler_training.py').read_text(encoding='utf-8')
check('source references art.langgraph.init_chat_model', 'init_chat_model' in rollout_src)
check('source references art.langgraph.wrap_rollout', 'wrap_rollout' in training_src or 'wrap_rollout' in rollout_src)
run_single = (ROOT/'scripts/gpu/run_02_train_policy_qlora_grpo.sh').read_text(encoding='utf-8')
check('GPU single-run includes official_art_ruler', 'official_art_ruler' in run_single)
check('GPU single-run has honest official preflight', 'art_availability' in run_single and 'Official ART/RULER unavailable' in run_single)
runall = (ROOT/'scripts/gpu/run_all_trackb_options_one_by_one.sh').read_text(encoding='utf-8')
check('GPU run-all includes official_art_ruler', 'official_art_ruler' in runall)
check('GPU run-all includes RULER checkpoint', 'trackb_official_art_ruler' in runall)
trackb = (ROOT/'src/ui/components/trainer_cards.py').read_text(encoding='utf-8')
check('Streamlit has Official ART + RULER card', 'Official ART + RULER' in trackb)
if not (status.get('art') and status.get('art_langgraph') and status.get('ruler')):
    print('Official ART/RULER unavailable.')
    print('Install:')
    print('uv pip install -U "openpipe-art[backend,langgraph]>=0.4.9"')
    # Not a failure: guarded optional integration must validate even when package is absent.
else:
    print('Official ART/RULER validation passed.')
if failures:
    print(f'Official ART/RULER validation FAILED: {len(failures)} issue(s).')
    sys.exit(1)

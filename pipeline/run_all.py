"""Run the three steps in order: universe -> download -> train and predict.

    python run_all.py              # first run: full history, full backtest (needs internet)
    python run_all.py --fast       # quicker backtest, slightly rougher numbers
    python run_all.py --daily      # daily refresh: fetch only the new days, redo the
                                   # slow backtest at most once a week, refresh predictions
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
data = here.parent / "data"
fast = ["--step", "126", "--max-train-rows", "800000", "--iters", "100"] if "--fast" in sys.argv else []
daily = "--daily" in sys.argv
steps = [
    ("01_universe.py", []),
    ("02_download.py", ["--incremental"] if daily else []),
    ("03_train_predict.py", fast + (["--eval-cache", str(data / "eval_cache.json"), "--eval-every", "7"] if daily else [])),
]
for script, extra in steps:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(here / script), *extra], check=True)
print("\nDone. Open web/index.html in your browser.")

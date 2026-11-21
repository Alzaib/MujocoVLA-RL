import sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from eval_bc import evaluate
from bc_model import load_policy

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50      # rollouts per instruction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, q_mean, q_std, vocab = load_policy(str(HERE / "bc_baseline.pt"), device)

    t0 = time.time()
    evaluate(model, q_mean, q_std, vocab, n_per_instruction=n, seed=123, device=device,
             out_dir=HERE / "results" / f"reproduce_n{n}",
             checkpoint="reinforcement_learning/baseline/bc_baseline.pt")
    secs = time.time() - t0
    print(f"{secs:.0f} s total, {secs / (6 * n):.1f} s per episode")
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from bc_model import load_policy

device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE / "bc_baseline.pt"), device)

print("device:", device)
print("parameters:", sum(p.numel() for p in model.parameters()))
print("vocab size:", len(vocab))
print("eval mode:", not model.training)
print("q_mean:", q_mean.cpu().numpy().round(2))
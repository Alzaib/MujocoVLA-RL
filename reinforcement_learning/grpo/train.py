"""train.py (GRPO) - residual actor trained with group-relative advantages. No critic."""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grpo_collect import collect_batch            # also sets up the other import paths
from grpo_update import make_batch, update_step

from env import PickPlaceEnv, sample_layout, BOX_COLORS, ZONE_COLORS
from bc_model import load_policy
from rollout import run_episode_recorded
from actor import ResidualActor, make_actor_fn

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--iters", type=int, default=500)
p.add_argument("--n_groups", type=int, default=2)       # scenes per iteration
p.add_argument("--group_size", type=int, default=8)     # episodes per scene
p.add_argument("--actor_lr", type=float, default=1e-3)
p.add_argument("--max_delta", type=float, default=0.03)
p.add_argument("--init_sigma", type=float, default=0.01)
p.add_argument("--val_every", type=int, default=25)
p.add_argument("--val_episodes", type=int, default=60)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

run_dir = HERE / "results" / args.run
run_dir.mkdir(parents=True, exist_ok=False)             # never overwrite a run
json.dump(vars(args), open(run_dir / "config.json", "w"), indent=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)                            # makes the sampled noise repeatable
model, q_mean, q_std, vocab = load_policy(str(HERE.parent / "baseline" / "bc_baseline.pt"), device)
model.eval()
for prm in model.parameters():
    prm.requires_grad_(False)                           # BC stays frozen
to_np = lambda x: torch.as_tensor(x).detach().cpu().numpy()
qm, qs = to_np(q_mean), to_np(q_std)

actor = ResidualActor(max_delta=args.max_delta, init_sigma=args.init_sigma)
actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
env = PickPlaceEnv()

train_rng = np.random.default_rng(2000 + args.seed)     # training scenes (not the test seed 123)
val_rng = np.random.default_rng(777)                    # the SAME validation set as the blackbox run
val_tasks = [(sample_layout(val_rng),
              BOX_COLORS[val_rng.integers(len(BOX_COLORS))],
              ZONE_COLORS[val_rng.integers(len(ZONE_COLORS))])
             for _ in range(args.val_episodes)]

def validate():
    """Noise-free actor on the fixed validation set -> (success rate, mean reward)."""
    wins, total = 0, 0.0
    for layout, box, zone in val_tasks:
        fn = make_actor_fn(actor, box, zone, qm, qs, explore=False)
        out = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone,
                                   device, actor_fn=fn)
        wins += out["success"]
        total += sum(out["rewards"])
    n = len(val_tasks)
    return wins / n, total / n


def save(tag):
    torch.save(actor.state_dict(), run_dir / f"actor_{tag}.pt")


train_f = open(run_dir / "train_log.csv", "w", newline="")
train_w = csv.writer(train_f)
train_w.writerow(["iter", "success", "episodes", "mean_reward", "mixed_groups",
                  "actor_loss", "sigma", "mean_abs_mu"])
val_f = open(run_dir / "val_log.csv", "w", newline="")
val_w = csv.writer(val_f)
val_w.writerow(["iter", "val_success", "val_mean_reward"])


def run_validation(it):
    s, r = validate()
    val_w.writerow([it, round(s, 4), round(r, 4)])
    val_f.flush()
    save(f"it{it:04d}")
    print(f"[val {it:4d}] success {s:.0%}  reward {r:.3f}", flush=True)


t0 = time.time()
run_validation(0)                                       # BC alone, before any learning
n_eps = args.n_groups * args.group_size

for it in range(1, args.iters + 1):
    # 1. collect: several groups, each on its own scene, with the actor adding sampled noise
    groups = collect_batch(env, model, q_mean, q_std, vocab, actor, qm, qs, train_rng,
                           args.n_groups, args.group_size, device)
    # 2. group advantages (the critic's replacement), then one actor update
    obs, delta, adv, mixed = make_batch(groups)
    info = update_step(actor, actor_opt, obs, delta, adv)

    # 3. log
    wins = int(sum(g["success"].sum() for g in groups))
    reward = float(np.mean(np.concatenate([g["ep_reward"] for g in groups])))
    n_mixed = round(mixed * args.n_groups)
    with torch.no_grad():
        sigma = actor.log_std.exp().mean().item()
        mu = actor.mean(obs).abs().mean().item()
    train_w.writerow([it, wins, n_eps, round(reward, 4), n_mixed,
                      round(info["actor_loss"], 5), round(sigma, 5), round(mu, 5)])
    train_f.flush()
    print(f"it {it:4d}/{args.iters} | batch {wins:2d}/{n_eps} reward {reward:.2f} | "
          f"mixed {n_mixed}/{args.n_groups} | sigma {sigma:.4f} | mean|mu| {mu:.4f} | "
          f"{time.time() - t0:.0f}s", flush=True)

    if it % args.val_every == 0 or it == args.iters:
        run_validation(it)

save("final")
train_f.close(); val_f.close()
print("done. saved to", run_dir)
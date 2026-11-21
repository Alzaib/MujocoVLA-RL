"""train.py - actor-critic training of the residual actor (black-box version)."""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv, sample_layout, BOX_COLORS, ZONE_COLORS
from bc_model import load_policy
from rollout import run_episode_recorded
from actor import ResidualActor, make_actor_fn
from critic import Critic
from rl_collect import collect_batch
from update import update_step

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)                 # name of the run folder
p.add_argument("--iters", type=int, default=500)
p.add_argument("--batch", type=int, default=16)        # episodes per iteration
p.add_argument("--actor_lr", type=float, default=3e-4)
p.add_argument("--critic_lr", type=float, default=1e-3)
p.add_argument("--max_delta", type=float, default=0.03)
p.add_argument("--init_sigma", type=float, default=0.005)
p.add_argument("--val_every", type=int, default=25)
p.add_argument("--val_episodes", type=int, default=60)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

# ---- run folder (never overwrite an old run) ------------------------------
run_dir = HERE / "results" / args.run
run_dir.mkdir(parents=True, exist_ok=False)
with open(run_dir / "config.json", "w") as f:
    json.dump(vars(args), f, indent=1)

# ---- setup ------------------------------------------------------------------
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE.parent / "baseline" / "bc_baseline.pt"), device)
for prm in model.parameters():
    prm.requires_grad_(False)                          # BC stays frozen
qm, qs = q_mean.cpu().numpy(), q_std.cpu().numpy()
env = PickPlaceEnv()

actor = ResidualActor(max_delta=args.max_delta, init_sigma=args.init_sigma)
critic = Critic()
actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

train_rng = np.random.default_rng(2000 + args.seed)    # training layouts (not the test seed 123)

# a fixed validation set (seed 777: not the training seed, not the test seed 123)
val_rng = np.random.default_rng(777)
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
    torch.save(critic.state_dict(), run_dir / f"critic_{tag}.pt")


# ---- logs -------------------------------------------------------------------
train_f = open(run_dir / "train_log.csv", "w", newline="")
train_w = csv.writer(train_f)
train_w.writerow(["iter", "success", "episodes", "mean_reward", "actor_loss",
                  "critic_loss", "sigma", "mean_abs_mu"])
val_f = open(run_dir / "val_log.csv", "w", newline="")
val_w = csv.writer(val_f)
val_w.writerow(["iter", "val_success", "val_mean_reward"])

t0 = time.time()
vs, vr = validate()                                    # iteration 0 = BC with a zero actor
val_w.writerow([0, vs, vr]); val_f.flush()
print(f"[val  0] success {vs:.0%}  reward {vr:.3f}   ({time.time() - t0:.0f}s)", flush=True)
save("it0000")

# ---- the training loop ------------------------------------------------------
for it in range(1, args.iters + 1):
    obs, delta, ret, stats = collect_batch(env, model, q_mean, q_std, vocab, actor, qm, qs,
                                           train_rng, args.batch, device)
    losses = update_step(actor, critic, actor_opt, critic_opt, obs, delta, ret)

    with torch.no_grad():
        sigma = actor.log_std.exp().mean().item()
        mean_abs_mu = actor.mean(obs).abs().mean().item()
    train_w.writerow([it, stats["success"], stats["episodes"], stats["mean_reward"],
                      round(losses["actor_loss"], 5), round(losses["critic_loss"], 5),
                      round(sigma, 5), round(mean_abs_mu, 5)])
    train_f.flush()
    print(f"it {it:4d}/{args.iters} | batch {stats['success']:2d}/{stats['episodes']} "
          f"reward {stats['mean_reward']:.2f} | sigma {sigma:.4f} | mean|mu| {mean_abs_mu:.4f} "
          f"| {time.time() - t0:.0f}s", flush=True)

    if it % args.val_every == 0 or it == args.iters:
        vs, vr = validate()
        val_w.writerow([it, vs, vr]); val_f.flush()
        print(f"[val {it:3d}] success {vs:.0%}  reward {vr:.3f}", flush=True)
        save(f"it{it:04d}")

save("final")
print("done. saved to", run_dir)
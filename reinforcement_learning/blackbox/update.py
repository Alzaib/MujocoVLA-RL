"""update.py - one learning step for the actor and the critic from one collected batch."""


def update_step(actor, critic, actor_opt, critic_opt, obs, delta, returns):
    """obs (N,20), delta (N,7), returns (N,): the three tables from collect_batch."""
    # 1. what the critic expected, and how far the real return was from that
    value = critic(obs)                                      # (N,)
    advantage = (returns - value).detach()                   # detach: no learning through this
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # 2. actor: make corrections with positive advantage more likely, negative less likely
    log_prob = actor.log_prob(obs, delta)                    # (N,)
    actor_loss = -(advantage * log_prob).mean()

    # 3. critic: move its prediction toward the real return
    critic_loss = ((value - returns) ** 2).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
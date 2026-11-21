"""reward.py - reward for one episode: 0.5 if the box is lifted, +1 if the episode succeeds."""

LIFT_Z = 0.05          # box centre above this (metres) counts as lifted; resting height is 0.02
LIFT_REWARD = 0.5
SUCCESS_REWARD = 1.0


def episode_rewards(lift_z, success):
    """lift_z: the target box's height at the end of step 2.
    success: whether the episode ended in success.
    Returns the 7 per-step rewards."""
    rewards = [0.0] * 7
    if lift_z > LIFT_Z:
        rewards[2] = LIFT_REWARD        # given right after the lift step
    if success:
        rewards[6] = SUCCESS_REWARD     # given at the end
    return rewards
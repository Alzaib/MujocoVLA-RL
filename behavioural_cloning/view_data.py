"""view_data.py: look at one episode (7 images) and a summary of a packed .npz file."""
import numpy as np
import matplotlib.pyplot as plt

STEP_NAMES = ["above_box", "grasp", "above_box", "above_drop", "release", "above_drop", "home"]


def show_episode(path, index=None):
    data = np.load(path)
    n = len(data["distance"])
    i = np.random.randint(n) if index is None else index

    images = data["images"][i]                  # (7, H, W, 3)
    finger = data["finger"][i]                  # (7,)

    fig, axes = plt.subplots(1, 7, figsize=(21, 3.8))
    for k, ax in enumerate(axes):
        ax.imshow(images[k])
        ax.set_title(f"step {k}: {STEP_NAMES[k]}\nfinger {finger[k]:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"episode {i} of {n}: {data['instruction'][i]}")
    plt.tight_layout()
    plt.show()


def summary(path):
    data = np.load(path)
    names, counts = np.unique(data["instruction"], return_counts=True)
    for name, c in zip(names, counts):
        print(f"{c:4d}  {name}")
    d = data["distance"] * 1000
    print(f"final box-zone distance (mm): min {d.min():.1f}, mean {d.mean():.1f}, max {d.max():.1f}")


if __name__ == "__main__":
    summary("data/val.npz")
    show_episode("data/val.npz")                # random episode
    # show_episode("data/val.npz", index=3)
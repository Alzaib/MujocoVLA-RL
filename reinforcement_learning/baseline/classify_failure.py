import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
eps = json.load(open(HERE / "results" / "diagnosis_data.json"))["episodes"]

LIFT_Z = 0.05     # box centre above this = in the air (resting height is 0.02)
MOVED = 0.02      # another box moved more than 2 cm = the wrong box was touched

CATS = ["wrong box moved", "never lifted", "dropped on the way", "wrong zone",
        "not resting (z)", "near miss (4-8 cm)", "far miss (>8 cm)"]


def classify(r):
    box, zone, steps = r["box"], r["zone"], r["boxes_after_step"]
    # 1. did another box move?
    for c, start in r["box_start"].items():
        if c != box:
            if np.linalg.norm(np.array(steps[6][c][:2]) - np.array(start[:2])) > MOVED:
                return "wrong box moved"
    # 2-3. was the target box lifted, and did it stay up until the zone?
    lifted2 = steps[2][box][2] > LIFT_Z
    lifted3 = steps[3][box][2] > LIFT_Z
    if not lifted2 and not lifted3:
        return "never lifted"
    if lifted2 and not lifted3:
        return "dropped on the way"
    # 4. did it end up in the other zone?
    final = np.array(steps[6][box][:2])
    other = [c for c in r["zones"] if c != zone][0]
    if np.linalg.norm(final - np.array(r["zones"][other][:2])) < 0.04:
        return "wrong zone"
    # 5. otherwise a placement problem
    d = r["distance"]
    if d < 0.04:
        return "not resting (z)"
    if d < 0.08:
        return "near miss (4-8 cm)"
    return "far miss (>8 cm)"


fails = [r for r in eps if not r["success"]]
print(f"{len(fails)} failures out of {len(eps)}\n")

total = Counter(classify(r) for r in fails)
for c in CATS:
    print(f"  {c:22s} {total[c]:4d}  ({total[c] / len(fails):.0%})")

print("\nper instruction:")
by = defaultdict(Counter)
for r in fails:
    by[(r["box"], r["zone"])][classify(r)] += 1
for (b, z), cnt in by.items():
    print(f"  {b}-{z}: " + ", ".join(f"{c} {cnt[c]}" for c in CATS if cnt[c]))

print("\nfailures that ended within 8 cm of the zone:",
      sum(r["distance"] < 0.08 for r in fails))
amb = [r for r in fails
       if 0.03 < max(r["boxes_after_step"][2][r["box"]][2],
                     r["boxes_after_step"][3][r["box"]][2]) < 0.09]
print("failures with an ambiguous lift (0.03-0.09 m):", len(amb))
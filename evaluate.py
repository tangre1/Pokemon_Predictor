"""Check how well the predictions match the winners in the dataset.

The data is from Kaggle's "Pokemon Challenge" by Weedle's Cave. There are about
50,000 battles in validation_data/, so this can run without an internet connection.

Run python evaluate.py for 2,000 test battles and 10,000 training battles.
For a bigger run, try python evaluate.py --battles 5000 --train 20000 --sims 50.
Use --no-ml to check just the speed rule and simulator.

This compares picking the faster Pokemon, the simulator by itself, and the
simulator with its probabilities adjusted using the training data. It also checks
gradient boosting with just stats and types, then with simulator results added.
All models use the same test battles. Test matchups stay out of training even
when the Pokemon are listed in reverse order. Draws count as half a win per side.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from predictor import (FEATURE_NAMES, HybridPredictor, SimCalibrator, feature_matrix,
                       load_dataset, to_pokemon)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """p is the first Pokemon's predicted win chance; y is 1 if it actually won."""
    pc = np.clip(p, 1e-3, 1 - 1e-3)
    pred = np.where(p > 0.5, 1, np.where(p < 0.5, 0, -1))  # exact ties count as wrong
    return {
        "accuracy": float(np.mean(pred == y)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append((f"{edges[b]:.1f}-{edges[b+1]:.1f}", int(m.sum()),
                         float(p[m].mean()), float(y[m].mean())))
    return pd.DataFrame(rows, columns=["bin", "n", "predicted", "actual"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=2000, help="held-out battles to score")
    ap.add_argument("--train", type=int, default=10000, help="battles used to train the ML models")
    ap.add_argument("--sims", type=int, default=50, help="Monte Carlo runs per matchup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ml", action="store_true")
    args = ap.parse_args()

    mons, combats, chart = load_dataset()
    test = combats.sample(n=min(args.battles, len(combats)), random_state=args.seed)

    # Keep test matchups out of training, including ones with the Pokemon swapped.
    test_pairs = set(zip(test["First_pokemon"], test["Second_pokemon"]))
    test_pairs |= {(b, a) for a, b in test_pairs}
    pool = combats.drop(test.index)
    pool = pool[[(a, b) not in test_pairs for a, b in zip(pool["First_pokemon"], pool["Second_pokemon"])]]
    train = pool.sample(n=min(args.train, len(pool)), random_state=args.seed)

    y = test["y"].to_numpy()
    print(f"{len(test):,} test battles | {len(train):,} train battles | "
          f"{args.sims} sims/matchup | first-pokemon win rate {y.mean():.3f}")

    def pairs(df):
        return [(to_pokemon(mons.loc[i]), to_pokemon(mons.loc[j]))
                for i, j in zip(df["First_pokemon"], df["Second_pokemon"])]

    print("Running simulator on test battles...")
    Xte = feature_matrix(pairs(test), chart, args.sims, args.seed)

    results: dict[str, np.ndarray] = {}
    sp_a = mons.loc[test["First_pokemon"], "Speed"].to_numpy()
    sp_b = mons.loc[test["Second_pokemon"], "Speed"].to_numpy()
    results["speed rule"] = np.where(sp_a > sp_b, 1.0, np.where(sp_a < sp_b, 0.0, 0.5))
    results["simulator (raw)"] = Xte[:, 0]

    if not args.no_ml:
        print("Running simulator on training battles...")
        Xtr = feature_matrix(pairs(train), chart, args.sims, args.seed)
        ytr = train["y"].to_numpy()

        results["simulator (calibrated)"] = SimCalibrator().fit(Xtr[:, 0], ytr).predict_proba(Xte[:, 0])

        no_sim = [i for i, n in enumerate(FEATURE_NAMES) if n not in ("sim_p", "sim_turns")]
        stats_only = HybridPredictor(chart, args.sims, args.seed).fit_features(Xtr[:, no_sim], ytr)
        results["stats only (GBM)"] = stats_only.predict_features(Xte[:, no_sim])

        hybrid = HybridPredictor(chart, args.sims, args.seed).fit_features(Xtr, ytr)
        results["hybrid (sim + stats)"] = hybrid.predict_features(Xte)

    print(f"\n{'model':<26}{'accuracy':>10}{'log-loss':>10}{'brier':>8}")
    for name, p in results.items():
        m = metrics(p, y)
        print(f"{name:<26}{m['accuracy']:>10.3f}{m['log_loss']:>10.3f}{m['brier']:>8.3f}")
    n = len(y)
    print(f"(accuracy standard error is about +/-{100 * (0.9 * 0.1 / n) ** 0.5:.1f} points at this sample size)")

    best = "hybrid (sim + stats)" if "hybrid (sim + stats)" in results else "simulator (raw)"
    print(f"\nCalibration of {best} (predicted vs actual P(first wins)):")
    print(calibration_table(results[best], y).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    p = results[best]
    miss = test.assign(p=p)
    miss = miss[((miss.p > 0.5) & (miss.y == 0)) | ((miss.p < 0.5) & (miss.y == 1))]
    miss = miss.assign(conf=(miss.p - 0.5).abs()).sort_values("conf", ascending=False).head(8)
    print(f"\nMost confident misses of {best}:")
    for _, r in miss.iterrows():
        a, b = mons.loc[int(r.First_pokemon), "Name"], mons.loc[int(r.Second_pokemon), "Name"]
        print(f"  {a} vs {b}: P(first)={r.p:.2f}, actual winner = {'first' if r.y else 'second'}")


if __name__ == "__main__":
    main()

"""Use the simulator results and Pokemon stats together to predict who wins.

The simulator alone gives 0% or 100% for about 95% of matchups and misses how
important speed is in the dataset. The model learns from the simulator's win
rate and battle length, plus stats and type matchups, instead of just trusting
the simulation.

By default, it trains on 10,000 battles from Kaggle's "Pokemon Challenge" in
validation_data/. The trained model stays in memory, not in a saved file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from battle import Pokemon, TypeChart, effectiveness, run_simulations
from data import _scale_hp, _scale_stat

DATA_DIR = Path(__file__).parent / "validation_data"
STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
FEATURE_NAMES = (
    ["sim_p", "sim_turns", "eff_a_on_b", "eff_b_on_a"]
    + [f"a_{s}" for s in STATS] + [f"b_{s}" for s in STATS] + [f"d_{s}" for s in STATS]
)


# loading the battle data
def load_dataset():
    """Load the Pokemon table, battles, and type chart, with Pokemon IDs starting at 1."""
    mons = pd.read_csv(DATA_DIR / "pokemon.csv")
    combats = pd.read_csv(DATA_DIR / "combats.csv")
    raw = json.loads((DATA_DIR / "typechart.json").read_text())
    chart = {t: {k: set(v) for k, v in rel.items()} for t, rel in raw.items()}
    mons.index = mons.index + 1  # battle IDs start at 1, not 0
    combats["y"] = (combats["Winner"] == combats["First_pokemon"]).astype(int)
    return mons, combats, chart


def to_pokemon(row) -> Pokemon:
    types = tuple(t.lower() for t in (row["Type 1"], row["Type 2"]) if isinstance(t, str))
    return Pokemon(
        name=str(row["Name"]), types=types,
        hp=_scale_hp(int(row["HP"])),
        attack=_scale_stat(int(row["Attack"])), defense=_scale_stat(int(row["Defense"])),
        sp_attack=_scale_stat(int(row["Sp. Atk"])), sp_defense=_scale_stat(int(row["Sp. Def"])),
        speed=_scale_stat(int(row["Speed"])),
    )


# inputs for the model
def pair_features(a: Pokemon, b: Pokemon, chart: TypeChart, sims: int = 50, seed: int = 0) -> np.ndarray:
    s = run_simulations(a, b, chart, n=sims, seed=seed)
    sim_p = (s.wins_a + 0.5 * s.draws) / s.n
    eff_ab = max(effectiveness(t, b.types, chart) for t in a.types)
    eff_ba = max(effectiveness(t, a.types, chart) for t in b.types)
    av = [getattr(a, k) for k in STATS]
    bv = [getattr(b, k) for k in STATS]
    return np.array([sim_p, float(np.mean(s.turns)), eff_ab, eff_ba]
                    + av + bv + [x - y for x, y in zip(av, bv)], dtype=float)


def feature_matrix(pairs, chart: TypeChart, sims: int = 50, seed: int = 0) -> np.ndarray:
    """Build the inputs for each Pokemon pair, reusing results for repeated pairs."""
    cache: dict[tuple[Pokemon, Pokemon], np.ndarray] = {}
    rows = []
    for a, b in pairs:
        if (a, b) not in cache:
            cache[(a, b)] = pair_features(a, b, chart, sims, seed)
        rows.append(cache[(a, b)])
    return np.vstack(rows)


# prediction models
class SimCalibrator:
    """Use logistic regression to adjust the simulator's overconfident win rates."""

    def __init__(self) -> None:
        self.lr = LogisticRegression()

    @staticmethod
    def _x(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, sim_p: np.ndarray, y: np.ndarray) -> "SimCalibrator":
        self.lr.fit(self._x(sim_p), y)
        return self

    def predict_proba(self, sim_p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._x(sim_p))[:, 1]


class HybridPredictor:
    def __init__(self, chart: TypeChart, sims: int = 50, seed: int = 0) -> None:
        self.chart, self.sims, self.seed = chart, sims, seed
        self.model = HistGradientBoostingClassifier(random_state=seed)

    def fit_features(self, X: np.ndarray, y: np.ndarray) -> "HybridPredictor":
        self.model.fit(X, y)
        return self

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, a: Pokemon, b: Pokemon) -> float:
        """Estimate the chance that a beats b."""
        X = pair_features(a, b, self.chart, self.sims, self.seed).reshape(1, -1)
        return float(self.predict_features(X)[0])

    def predict_symmetric(self, a: Pokemon, b: Pokemon) -> float:
        """Try both orders and average so being listed first doesn't give an advantage."""
        return 0.5 * (self.predict(a, b) + (1.0 - self.predict(b, a)))


_default: HybridPredictor | None = None


def is_trained() -> bool:
    return _default is not None


def get_default_predictor(n_train: int = 10000, sims: int = 30, seed: int = 0) -> HybridPredictor:
    """Train on validation_data/ the first time, then reuse the model."""
    global _default
    if _default is None:
        mons, combats, chart = load_dataset()
        train = combats.sample(n=min(n_train, len(combats)), random_state=seed)
        pairs = [(to_pokemon(mons.loc[i]), to_pokemon(mons.loc[j]))
                 for i, j in zip(train["First_pokemon"], train["Second_pokemon"])]
        X = feature_matrix(pairs, chart, sims, seed)
        _default = HybridPredictor(chart, sims, seed).fit_features(X, train["y"].to_numpy())
    return _default

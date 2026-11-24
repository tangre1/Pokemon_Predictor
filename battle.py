"""Simulate battles using Pokemon stats and type matchups.

For now, everyone is level 50 and gets one attack per turn with 80 base power.
Each Pokemon picks its own type with the best matchup, so it always gets STAB.
It uses physical or special attacks based on which attack stat is higher.
The faster Pokemon goes first, with a random pick if their speeds match.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

LEVEL = 50
MOVE_POWER = 80
STAB = 1.5
MAX_TURNS = 100

# Each attack type has sets of types it does double, half, or no damage to.
TypeChart = dict[str, dict[str, set[str]]]


@dataclass(frozen=True)
class Pokemon:
    name: str
    types: tuple[str, ...]
    hp: int
    attack: int
    defense: int
    sp_attack: int
    sp_defense: int
    speed: int


@dataclass
class BattleResult:
    winner: str | None  # None if nobody wins
    winner_idx: int | None  # 0 for first, 1 for second, None for a draw, even with matching names
    turns: int
    hp_left_pct: float  # fraction of the winner's full HP left, not a percentage
    history: list[tuple[int, int]] = field(default_factory=list)  # both HP values at the start and after each turn


def effectiveness(move_type: str, defender_types: tuple[str, ...], chart: TypeChart) -> float:
    mult = 1.0
    rel = chart[move_type]
    for t in defender_types:
        if t in rel["no"]:
            mult *= 0.0
        elif t in rel["double"]:
            mult *= 2.0
        elif t in rel["half"]:
            mult *= 0.5
    return mult


def best_attack(attacker: Pokemon, defender: Pokemon, chart: TypeChart) -> tuple[float, int, int]:
    """Pick the best attack and return its multiplier (with STAB), attack, and defense stats."""
    eff = max(effectiveness(t, defender.types, chart) for t in attacker.types)
    if attacker.attack >= attacker.sp_attack:
        return eff * STAB, attacker.attack, defender.defense
    return eff * STAB, attacker.sp_attack, defender.sp_defense


def calc_damage(attacker: Pokemon, defender: Pokemon, chart: TypeChart, rng: random.Random) -> int:
    mult, atk, dfn = best_attack(attacker, defender, chart)
    if mult == 0:
        return 0
    base = ((2 * LEVEL / 5 + 2) * MOVE_POWER * atk / dfn) / 50 + 2
    return max(1, int(base * mult * rng.uniform(0.85, 1.0)))


def simulate_battle(a: Pokemon, b: Pokemon, chart: TypeChart, rng: random.Random) -> BattleResult:
    history = [(a.hp, b.hp)]
    turns = 0
    fighters = [a, b]  # use positions so two Pokemon with the same name still work
    hps = [a.hp, b.hp]

    while turns < MAX_TURNS and hps[0] > 0 and hps[1] > 0:
        turns += 1
        if a.speed != b.speed:
            order = [0, 1] if a.speed > b.speed else [1, 0]
        else:
            order = [0, 1] if rng.random() < 0.5 else [1, 0]
        for i in order:
            j = 1 - i
            if hps[i] <= 0:
                break
            hps[j] = max(0, hps[j] - calc_damage(fighters[i], fighters[j], chart, rng))
            if hps[j] <= 0:
                break
        history.append((hps[0], hps[1]))

    if hps[0] > 0 and hps[1] > 0:
        return BattleResult(None, None, turns, 0.0, history)
    w = 0 if hps[0] > 0 else 1
    return BattleResult(fighters[w].name, w, turns, hps[w] / fighters[w].hp, history)


@dataclass
class SimulationSummary:
    a: Pokemon
    b: Pokemon
    n: int
    wins_a: int
    wins_b: int
    draws: int
    turns: list[int]
    hp_left_a: list[float]  # fraction of HP left each time A wins
    hp_left_b: list[float]
    sample_history: list[tuple[int, int]]


def run_simulations(a: Pokemon, b: Pokemon, chart: TypeChart,
                    n: int = 2000, seed: int | None = None) -> SimulationSummary:
    rng = random.Random(seed)
    wins_a = wins_b = draws = 0
    turns, left_a, left_b = [], [], []
    sample: list[tuple[int, int]] = []
    for k in range(n):
        r = simulate_battle(a, b, chart, rng)
        if k == 0:
            sample = r.history
        turns.append(r.turns)
        if r.winner_idx is None:
            draws += 1
        elif r.winner_idx == 0:
            wins_a += 1
            left_a.append(r.hp_left_pct)
        else:
            wins_b += 1
            left_b.append(r.hp_left_pct)
    return SimulationSummary(a, b, n, wins_a, wins_b, draws, turns, left_a, left_b, sample)

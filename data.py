"""Data layer: pulls Pokemon and type-chart data from PokeAPI (https://pokeapi.co).

Everything is cached in memory only (the proposal says no loading/saving to disk).
"""
from __future__ import annotations

import requests

from battle import LEVEL, Pokemon, TypeChart

BASE = "https://pokeapi.co/api/v2"
BATTLE_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy",
]

_cache: dict[str, dict] = {}
_type_chart: TypeChart | None = None


def _get(url: str) -> dict:
    if url not in _cache:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        _cache[url] = resp.json()
    return _cache[url]


def get_pokemon_names() -> list[str]:
    data = _get(f"{BASE}/pokemon?limit=2000")
    return sorted(p["name"] for p in data["results"])


def _scale_hp(base: int) -> int:
    return (2 * base + 31) * LEVEL // 100 + LEVEL + 10


def _scale_stat(base: int) -> int:
    return (2 * base + 31) * LEVEL // 100 + 5


def get_pokemon(name: str) -> Pokemon:
    data = _get(f"{BASE}/pokemon/{name.lower().strip()}")
    s = {x["stat"]["name"]: x["base_stat"] for x in data["stats"]}
    return Pokemon(
        name=data["name"],
        types=tuple(t["type"]["name"] for t in data["types"]),
        hp=_scale_hp(s["hp"]),
        attack=_scale_stat(s["attack"]),
        defense=_scale_stat(s["defense"]),
        sp_attack=_scale_stat(s["special-attack"]),
        sp_defense=_scale_stat(s["special-defense"]),
        speed=_scale_stat(s["speed"]),
    )


def get_type_chart() -> TypeChart:
    global _type_chart
    if _type_chart is None:
        chart: TypeChart = {}
        for t in BATTLE_TYPES:
            rel = _get(f"{BASE}/type/{t}")["damage_relations"]
            chart[t] = {
                "double": {x["name"] for x in rel["double_damage_to"]},
                "half": {x["name"] for x in rel["half_damage_to"]},
                "no": {x["name"] for x in rel["no_damage_to"]},
            }
        _type_chart = chart
    return _type_chart

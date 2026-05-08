"""Tracks birth, death, and ancestry of every agent in the simulation."""

import json
import os
from typing import Dict, Any

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_agents")
LINEAGE_FILE = os.path.join(MEMORY_DIR, "lineage.json")
STATS_FILE = os.path.join(MEMORY_DIR, "stats.json")


class LineageTracker:
    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._lineage: Dict[str, Any] = {}
        self._dead_ids: set = set()
        self._stats: list = []

    def register_birth(self, agent, step: int) -> None:
        self._lineage[agent.id] = {
            "generation": agent.generation,
            "parent_ids": agent.parent_ids,
            "born_step": step,
            "died_step": None,
            "cause": None,
            "fitness": None,
        }

    def register_death(self, agent, step: int, cause: str = "unknown") -> None:
        if agent.id in self._lineage:
            self._lineage[agent.id]["died_step"] = step
            self._lineage[agent.id]["cause"] = cause
            self._lineage[agent.id]["fitness"] = float(agent.fitness)
        self._dead_ids.add(agent.id)

    def is_registered_dead(self, agent_id: str) -> bool:
        return agent_id in self._dead_ids

    def record_stats(self, stats: dict) -> None:
        self._stats.append(stats)
        if len(self._stats) % 50 == 0:
            self.flush()

    def flush(self) -> None:
        try:
            with open(LINEAGE_FILE, "w") as f:
                json.dump(self._lineage, f, indent=2)
            with open(STATS_FILE, "w") as f:
                json.dump(self._stats[-500:], f, indent=2)  # keep last 500 records
        except IOError:
            pass

    def get_max_generation(self) -> int:
        if not self._lineage:
            return 0
        return max(v["generation"] for v in self._lineage.values())

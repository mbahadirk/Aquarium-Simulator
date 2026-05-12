"""Persists the best-performing agents and their genomes to JSON."""

import json
import os
from typing import List


SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_agents"))
BEST_FILE = os.path.join(SAVE_DIR, "best_agents.json")


class AgentMemory:
    def __init__(self, config: dict):
        mc = config.get("memory", {})
        self._top_n: int = mc.get("save_top_n", 10)
        os.makedirs(SAVE_DIR, exist_ok=True)
        self._best: List[dict] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_best_agents(self, agents: list, step: int) -> None:
        """Save the top-N agents by fitness, merging with previously saved."""
        records = [
            {
                "id": a.id,
                "lineage_id": a.lineage_id,
                "fitness": float(a.fitness),
                "generation": a.generation,
                "weights": a.genome.to_list(),
                "age": a.age,
                "food_eaten": a.total_food_eaten,
                "children": a.children_count,
                "saved_step": step,
            }
            for a in agents
        ]
        combined = self._best + records
        combined.sort(key=lambda r: r["fitness"], reverse=True)
        self._best = combined[: self._top_n]
        self._persist()

    def load_best_genomes(self) -> List[dict]:
        """Return list of {weights, generation, lineage_id} for seeding / recovery."""
        return [
            {
                "weights":    r["weights"],
                "generation": r["generation"],
                "lineage_id": r.get("lineage_id"),
            }
            for r in self._best
        ]

    def load_best_genomes_full(self) -> List[dict]:
        """Return full saved records (includes fitness, name, etc.)."""
        return list(self._best)

    def save_named_agent(self, agent, name: str, step: int) -> None:
        """Force-save a single agent with a custom name regardless of fitness rank."""
        record = {
            "id":         agent.id,
            "name":       name,
            "lineage_id": agent.lineage_id,
            "fitness":    float(agent.fitness),
            "generation": agent.generation,
            "weights":    agent.genome.to_list(),
            "age":        agent.age,
            "food_eaten": agent.total_food_eaten,
            "children":   agent.children_count,
            "saved_step": step,
        }
        # Replace existing entry with same id or append
        self._best = [r for r in self._best if r["id"] != agent.id]
        self._best.append(record)
        self._best.sort(key=lambda r: r["fitness"], reverse=True)
        self._persist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> List[dict]:
        if os.path.exists(BEST_FILE):
            try:
                with open(BEST_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _persist(self) -> None:
        with open(BEST_FILE, "w") as f:
            json.dump(self._best, f, indent=2)

"""Persists the best-performing agents and their genomes to JSON."""

import json
import os
import tempfile
from typing import List


SAVE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_agents"))
BEST_FILE  = os.path.join(SAVE_DIR, "best_agents.json")
NAMED_FILE = os.path.join(SAVE_DIR, "named_agents.json")


class AgentMemory:
    def __init__(self, config: dict):
        mc = config.get("memory", {})
        self._top_n: int = mc.get("save_top_n", 10)
        os.makedirs(SAVE_DIR, exist_ok=True)
        self._best:  List[dict] = self._load(BEST_FILE)
        self._named: List[dict] = self._load(NAMED_FILE)  # user-pinned, never evicted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_best_agents(self, agents: list, step: int) -> None:
        """Auto-save top-N agents by fitness. Does not touch named agents."""
        named_ids = {r["id"] for r in self._named}
        records = [
            {
                "id":         a.id,
                "lineage_id": a.lineage_id,
                "fitness":    float(a.fitness),
                "generation": a.generation,
                "weights":    a.genome.to_list(),
                "age":        a.age,
                "food_eaten": a.total_food_eaten,
                "children":   a.children_count,
                "saved_step": step,
            }
            for a in agents
            if a.id not in named_ids        # don't duplicate named agents here
        ]
        combined = self._best + records
        combined.sort(key=lambda r: r["fitness"], reverse=True)
        # Deduplicate by id (keep highest fitness per id)
        seen: set = set()
        deduped = []
        for r in combined:
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)
        self._best = deduped[: self._top_n]
        self._persist(self._best, BEST_FILE)

    def save_named_agent(self, agent, name: str, step: int) -> None:
        """Pin an agent by name — persisted forever, never evicted by auto-save."""
        record = {
            "id":         agent.id,
            "name":       name or agent.id[:8],
            "lineage_id": agent.lineage_id,
            "fitness":    float(agent.fitness),
            "generation": agent.generation,
            "weights":    agent.genome.to_list(),
            "age":        agent.age,
            "food_eaten": agent.total_food_eaten,
            "children":   agent.children_count,
            "saved_step": step,
        }
        # Replace existing entry with same id
        self._named = [r for r in self._named if r["id"] != agent.id]
        self._named.append(record)
        self._named.sort(key=lambda r: r["fitness"], reverse=True)
        # Also remove from auto-best if present there
        self._best = [r for r in self._best if r["id"] != agent.id]
        self._persist(self._named, NAMED_FILE)
        self._persist(self._best, BEST_FILE)

    def delete_named_agent(self, agent_id: str) -> None:
        self._named = [r for r in self._named if r["id"] != agent_id]
        self._persist(self._named, NAMED_FILE)

    def load_best_genomes(self) -> List[dict]:
        """Return {weights, generation, lineage_id} list for seeding / recovery."""
        all_records = self._named + self._best
        seen: set = set()
        result = []
        for r in all_records:
            if r["id"] not in seen:
                seen.add(r["id"])
                result.append({
                    "weights":    r["weights"],
                    "generation": r["generation"],
                    "lineage_id": r.get("lineage_id"),
                })
        return result

    def load_best_genomes_full(self) -> List[dict]:
        """Return full records: named agents first (pinned), then auto-best."""
        seen: set = set()
        result = []
        for r in self._named + self._best:
            if r["id"] not in seen:
                seen.add(r["id"])
                result.append(r)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> List[dict]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _persist(self, data: list, path: str) -> None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=SAVE_DIR, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            try:
                os.remove(path)
            except OSError:
                pass
            os.rename(tmp_path, path)
        except (OSError, PermissionError):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

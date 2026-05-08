import csv
import os
from typing import IO


class SimLogger:
    _FIELDS = [
        "step", "population", "food_count", "predator_count",
        "avg_energy", "avg_fitness", "max_generation",
    ]

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file: IO = open(log_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS, extrasaction="ignore")
        self._writer.writeheader()

    def log(self, step: int, stats: dict) -> None:
        row = {"step": step, **stats}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

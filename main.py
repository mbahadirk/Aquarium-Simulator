"""Entry point for the Aquarium Natural Selection Simulation."""

import os
import sys

import yaml

from aquarium.environment import AquariumEnvironment
from core.simulation_engine import SimulationEngine
from memory.agent_memory import AgentMemory
from memory.lineage_tracker import LineageTracker
from sim_logging.sim_logger import SimLogger
from visualization.renderer import Renderer


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    # ------------------------------------------------------------------ config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    # ------------------------------------------------------------------ memory
    memory = AgentMemory(config)
    lineage = LineageTracker()

    # ------------------------------------------------------------------ environment
    env = AquariumEnvironment(config, memory_manager=memory, lineage_tracker=lineage)

    # Seed initial population (from saved best agents if available)
    saved_genomes = []
    if config.get("simulation", {}).get("seed_from_saved", True):
        saved_genomes = memory.load_best_genomes()

    env.seed_population(saved_genomes or None)

    # ------------------------------------------------------------------ subsystems
    wc = config.get("world", {})
    renderer = Renderer(wc.get("width", 1280), wc.get("height", 720))

    log_path = config.get("logging", {}).get("log_file", "sim_logging/sim_log.csv")
    log_path = os.path.join(os.path.dirname(__file__), log_path)
    logger = SimLogger(log_path)

    # ------------------------------------------------------------------ engine
    engine = SimulationEngine(env, renderer=renderer, logger=logger, config=config)

    print("Aquarium Simulation started.")
    print("  SPACE  → pause / resume")
    print("  ESC    → quit")
    print(f"  Population: {config['agent']['initial_count']} agents")
    print()

    try:
        engine.run()
    except KeyboardInterrupt:
        engine.stop()

    lineage.flush()

    print("Simulation ended.")
    print(f"  Total steps: {engine.step_count}")
    print(f"  Max generation reached: {lineage.get_max_generation()}")
    print(f"  Best agents saved to:   saved_agents/best_agents.json")
    print(f"  Log saved to:           {log_path}")


if __name__ == "__main__":
    # Ensure project root is on path regardless of CWD
    sys.path.insert(0, os.path.dirname(__file__))
    main()

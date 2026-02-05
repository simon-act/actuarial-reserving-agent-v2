from pathlib import Path

# Paths relative to project root (assuming execution from root)
ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs" / "agent_runs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

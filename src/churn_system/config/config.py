from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "settings.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

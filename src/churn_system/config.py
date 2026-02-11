import yaml
from pathlib import Path

CONFIG_PATH = Path("config/settings.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
    


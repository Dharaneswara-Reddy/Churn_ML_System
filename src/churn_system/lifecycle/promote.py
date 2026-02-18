import shutil
from pathlib import Path
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__,CONFIG["logging"]["lifecycle"])


def promote_model(version: str):
    """
    Promote a trained model version to production.
    
    Parameters
        version : str
            Name of the model directory inside `models/experiments/`.
            Example: "churn_model_v1"
        
        1. Checks whether the requested experiment exists.
        2. Removes the existing production model (if any).
        3. Copies the selected experiment infor production.
        4. Makes the promoted model the one used by the API. 
        
        Note : 
            if the requested model version does not exist it raises ValueError.
            
        The API Always loads models from the production directory, never directly from experiments. 
    """
    experiments_dir = Path("models/experiments")
    production_dir = Path("models/production")
    
    source = experiments_dir / version
    target = production_dir / "current"
    
    if not source.exists():
        raise ValueError(f"Model version {version} does not exit.")
    
    if target.exists():
        shutil.rmtree(target)
        
    shutil.copytree(source, target)
    print(f"Model {version} promoted to production.")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: Python -m churn_system.promote <version>")
        sys.exit(1)
        
    promote_model(sys.argv[1])
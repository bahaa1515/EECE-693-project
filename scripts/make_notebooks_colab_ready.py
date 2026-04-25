from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

BOOTSTRAP_IMPORTS = """from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/bahaa1515/EECE-693-project.git"
COLAB_REPO_DIR = Path("/content/EECE-693-project")

def find_project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "src" / "config.py").exists():
            return candidate
    if COLAB_REPO_DIR.exists() and (COLAB_REPO_DIR / "src" / "config.py").exists():
        return COLAB_REPO_DIR
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise FileNotFoundError(
            "Could not find the project root. Run this notebook from the repo root, "
            "from the notebooks folder, or clone the repo in Colab first."
        ) from exc
    subprocess.run(["git", "clone", REPO_URL, str(COLAB_REPO_DIR)], check=True)
    return COLAB_REPO_DIR

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root: {PROJECT_ROOT}")
"""


REPLACEMENTS = {
    "01_data_understanding.ipynb": BOOTSTRAP_IMPORTS
    + """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import SMARTWATCH_FILES, PATIENT_INFO_FILE, WEEKLY_FILE, OUTPUT_TABLES
""",
    "02_smartwatch_cleaning.ipynb": BOOTSTRAP_IMPORTS
    + """
import pandas as pd
import numpy as np

from src.config import SMARTWATCH_FILES, DATA_INTERIM, OUTPUT_TABLES
""",
    "03_windowing_and_features.ipynb": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
"""
    + BOOTSTRAP_IMPORTS
    + """
from src.config import DATA_INTERIM, DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_FIGURES
""",
    "04_label_preparation.ipynb": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
    + BOOTSTRAP_IMPORTS
    + """
from src.config import DATA_RAW, DATA_PROCESSED, OUTPUT_TABLES
""",
    "05_baseline_model_local_or_colab.ipynb": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
"""
    + BOOTSTRAP_IMPORTS
    + """
from src.config import DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_FIGURES

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("XGBoost available")
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed - skipping")
""",
    "06_improved_baselines_tier1.ipynb": """# Notebook 06 - Improved Tier-1 Baselines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")

"""
    + BOOTSTRAP_IMPORTS
    + """
from src.config import DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_FIGURES

print("Imports OK | XGBoost available:", HAS_XGB)
""",
    "07_deep_learning_tier2.ipynb": BOOTSTRAP_IMPORTS
    + """
from src.deep_learning import run_deep_learning_experiment

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "PyTorch is not installed in this runtime. For VS Code Colab, connect "
        "to a Colab GPU runtime or run notebooks/08_colab_deep_learning_runner.ipynb. "
        "For local CPU testing, install torch first."
    ) from exc

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
""",
}


def replace_first_code_cell(path: Path, source: str) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first_code = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    )
    notebook["cells"][first_code]["source"] = [
        f"{line}\n" for line in source.rstrip().split("\n")
    ]
    notebook["cells"][first_code]["execution_count"] = None
    notebook["cells"][first_code]["outputs"] = []
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for notebook_name, source in REPLACEMENTS.items():
        replace_first_code_cell(NOTEBOOK_DIR / notebook_name, source)
        print(f"Updated {notebook_name}")


if __name__ == "__main__":
    main()

# Running the Deep-Learning Reproduction on Google Colab

The Tier-2 GRU/CNN run can be heavy on a local CPU because it builds 24-hour minute-level sequences and trains neural networks. Colab is useful because it can provide a free GPU runtime and usually already includes PyTorch.

If you are using the VS Code Colab extension, also see `VSCODE_COLAB_WORKFLOW.md`.
For the corrected v2 event-episode metric-selection protocol, use
`V2_METRIC_PROTOCOL_RUNBOOK.md` and `scripts/v2/run_metric_protocol.py`.

## Recommended Colab Runtime

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Go to `Runtime` > `Change runtime type`.
4. Set `Hardware accelerator` to `GPU`.
5. Click `Save`.

## Option A: Clone the GitHub Repository

Use this if the GitHub repo is public or Colab can access it.

```python
from pathlib import Path

REPO_URL = "https://github.com/bahaa1515/EECE-693-project.git"
REPO_DIR = Path("/content/EECE-693-project")

if REPO_DIR.exists():
    %cd /content/EECE-693-project
    !git pull
else:
    %cd /content
    !git clone {REPO_URL} EECE-693-project
    %cd /content/EECE-693-project
```

Install the Colab dependencies. This file excludes `torch` because Colab usually provides it already.

```python
!pip install -r requirements-colab.txt
```

Confirm GPU/PyTorch:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

Run the full reproduction pipeline (canonical numbers in the report):

```python
!python scripts/run_full_pipeline.py --full
```

## Corrected Event-Episode Pipeline

For the corrected questionnaire-only labeling strategy, put the full AAMOS CSV
folder in Google Drive, then point the project at it:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Example environment setup:

```python
import os
os.environ["AAMOS_DATA_DIR"] = "/content/drive/MyDrive/AAMOS/dataset"
os.environ["AAMOS_OUTPUT_DIR"] = "/content/drive/MyDrive/AAMOS/outputs"
```

Run the corrected label/sample sensitivity grid:

```python
!python scripts/run_questionnaire_event_pipeline.py --skip-features
```

Run a full sensor-feature + tabular-model smoke experiment:

```python
!python scripts/run_questionnaire_event_pipeline.py --thresholds 3 --input-lengths 7 --washouts 7
```

Run more combinations when the smoke run is clean:

```python
!python scripts/run_questionnaire_event_pipeline.py --train-all
```

The corrected pipeline uses questionnaires to create labels and uses
smartwatch, smart inhaler, peak-flow, environment, and patient-info data only as
prediction features before event onset.

The script writes:

- `outputs/tables/deep_learning_reproduced_results.csv`
- `outputs/tables/deep_learning_report_comparison.csv`
- model training histories in `outputs/tables/`

Display the comparison:

```python
import pandas as pd
pd.read_csv("outputs/tables/deep_learning_report_comparison.csv")
```

## Option B: Upload the Project Zip

Use this if the repo is private or cloning asks for credentials.

1. Zip the project folder, excluding `.venv/`.
2. Upload it into Colab from the left sidebar.
3. Unzip and enter the folder:

```python
!unzip EECE-693-project.zip
%cd EECE-693-project
!pip install -r requirements-colab.txt
!python -m src.deep_learning --epochs 20 --batch-size 64 --patience 5
```

## Saving Results Back

Download the generated CSV files from Colab, or mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then copy results:

```python
!cp outputs/tables/deep_learning_reproduced_results.csv /content/drive/MyDrive/
!cp outputs/tables/deep_learning_report_comparison.csv /content/drive/MyDrive/
```

## When To Use Colab Instead Of Local

Use Colab if:

- installing `torch` locally is slow or unstable
- training on CPU takes more than a few minutes
- you want faster GRU/CNN experimentation
- you want to test bigger models, longer epochs, or Transformer/TCN models

Use local `.venv` if:

- you are running cleaning, labeling, feature engineering, or tabular baselines
- you only need quick code checks
- you want to edit/debug the pipeline before pushing to GitHub

## VS Code Colab Extension Notes

The VS Code Colab extension edits notebooks from VS Code, but the notebook code
runs in Colab's remote runtime. The remote runtime starts without your local
repo files, so the runner notebook clones or pulls the GitHub repo under
`/content/EECE-693-project`.

The usual loop is:

1. Edit locally with VS Code/Codex.
2. Commit and push to GitHub.
3. Run `notebooks/08_colab_deep_learning_runner.ipynb` on a Colab GPU runtime.
4. Download or copy back the generated CSV result files.

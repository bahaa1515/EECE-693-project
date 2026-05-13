# VS Code + Codex + Colab Workflow

This project is set up for a split workflow:

- Use VS Code and Codex for editing, refactoring, documentation, commits, and pushes.
- Use a local `.venv` for lightweight checks, tabular models, and quick script validation.
- Use the VS Code Colab extension or Google Colab directly for GPU-heavy deep-learning runs.

## Mental Model

When a notebook is connected to a Colab runtime from VS Code, the code does not run on your local laptop. It runs on Colab's remote Linux machine. That means local files in VS Code are not automatically available inside the Colab runtime.

For this reason, the Colab runner notebook clones or pulls the GitHub repo into `/content/EECE-693-project` before running experiments.

## Normal Development Loop

1. Edit code locally in VS Code.
2. Ask Codex to make improvements in the repo.
3. Run quick checks locally when possible:

```powershell
.\.venv\Scripts\python.exe -m compileall src
```

4. Commit and push changes to GitHub.
5. Open `notebooks/08_colab_deep_learning_runner.ipynb` in VS Code.
6. Connect the notebook to a Colab GPU runtime.
7. Run the notebook cells. The first cell will clone or pull the latest GitHub code.
8. Inspect:

```text
outputs/tables/deep_learning_reproduced_results.csv
outputs/tables/deep_learning_report_comparison.csv
```

## Important Rule

Push before running Colab.

Colab pulls from GitHub. If the newest local edits are not pushed, Colab will run old code.

## Local Setup

Create a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If installing heavy packages such as `torch` or `xgboost` is slow or unstable locally, skip the local heavy install and use Colab for the deep-learning run.

## Colab Runner

The recommended GPU runner is:

```text
notebooks/08_colab_deep_learning_runner.ipynb
```

It performs:

1. Clone or pull `https://github.com/bahaa1515/EECE-693-project.git`.
2. Install `requirements-colab.txt`.
3. Confirm GPU availability.
4. Run the full reproduction pipeline (canonical numbers in the report):

```bash
python scripts/run_full_pipeline.py --full
```

5. Display reproduced metrics and comparison against the report table.

## Bringing Results Back

After the Colab run, download these generated files or copy their contents back into the local repo:

```text
outputs/tables/headline_results.csv
outputs/tables/experiment_comparison.csv
```

If the results are important for the report, commit them after review.

## Troubleshooting

If Colab says `ModuleNotFoundError: src`, make sure the notebook has run the clone/pull cell and that the current directory is:

```text
/content/EECE-693-project
```

If Colab says CUDA is unavailable, go to `Runtime` > `Change runtime type` and select `GPU`.

If `git pull` says local files would be overwritten, restart the Colab runtime and run from a clean clone.

# Titanic survivor prediction

Engineers categorical passenger features and compares classical models and CatBoost to produce competition predictions.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [kaggle_titanic_competition_submission.ipynb](kaggle_titanic_competition_submission.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `train.csv`
- `test.csv`
- `gender_submission.csv`

## Outputs

Model comparison tables, feature-importance plots, and catboost_submission.csv.

## Implementation and limitations

Use the Kaggle Titanic CSV schema. The bundled kaggle_titanic_submission_mansib_mursalin.csv is a historical result, not training data. fit_ml_algo reports training accuracy and cross-validation accuracy separately. Full CatBoost training was not rerun.

Reusable model/analysis functions live in `experiment_kaggle_titanic_competition_submission.py`. Notebooks retain the experiment sequence, training, and visual narrative.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.

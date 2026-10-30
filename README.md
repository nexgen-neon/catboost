# Student Outcome Prediction with CatBoost

A beginner-friendly machine learning project that uses student information
to predict **Pass or Fail**, along with an estimated probability of passing.

The model is trained in Google Colab, saved, and loaded locally for predictions.
An HTML report explains the project and its example results.

## What the Model Uses

| Input | Type | Description |
|---|---|---|
| study_hours | Numerical | Time spent studying |
| attendance | Numerical | Attendance percentage |
| learning_method | Categorical | Classroom, Online, or Self Study |
| internet_access | Categorical | Yes or No |

The training target is `passed`: `1` means Pass and `0` means Fail.

All four inputs are used together to make predictions.

## How CatBoost Works

CatBoost combines many small decision trees. Each tree asks questions
about the inputs, and successive trees help reduce the model’s errors.

During training, it learns from student records with known outcomes.
For new students, it combines what it learned to produce:

- A predicted outcome: **Pass or Fail**
- An estimated probability of passing

CatBoost has built-in support for categorical inputs. By identifying
the categorical columns during training, we can use values such as
“Online” and “Classroom” without manually assigning them a numerical ranking.

Its categorical handling is useful, but it does not guarantee better
performance than every other model.

## Dataset

This demonstration uses **1,000 artificially generated student records**.

| Split | Records | Purpose |
|---|---:|---|
| Training | 600 | Learn patterns |
| Validation | 200 | Monitor training and support early stopping |
| Test | 200 | Evaluate predictions on unseen records |

The outcomes were generated using chosen rules and randomness.
They do not represent real student performance.

## Example Predictions

The following results were reported from the Colab run:

| Study Hours | Attendance | Learning Method | Internet Access | Prediction | Pass Probability |
|---:|---:|---|---|---|---:|
| 6.5 | 92% | Classroom | Yes | Pass | 96.2% |
| 1.0 | 50% | Self Study | No | Fail | 3.8% |
| 4.0 | 75% | Online | Yes | Pass | 68.4% |

A pass probability is the model’s estimate—not an exam score
or a measure of overall model accuracy.

The actual outcomes for these three example students are not available.

## Project Structure

Place the files as follows:

    catboost-local/
    ├── src/
    │   └── catboost_local/
    │       ├── predict.py
    │       └── student_catboost.cbm
    ├── catboost_report.html
    ├── README.md
    ├── pyproject.toml
    └── uv.lock

| File | Purpose |
|---|---|
| predict.py | Loads the saved model and predicts student outcomes |
| student_catboost.cbm | Trained CatBoost model exported from Colab |
| catboost_report.html | Standalone visual project report |
| pyproject.toml | Project configuration and dependencies |
| uv.lock | Locked dependency versions |

## Local Setup

Requirements:

- Python compatible with CatBoost; this demo uses Python 3.12
- uv
- The trained `student_catboost.cbm` file

Open PowerShell in the project folder:

    cd C:\Users\kaush\catboost-local

If the project already contains `pyproject.toml` and `uv.lock`,
install its dependencies:

    uv sync

If CatBoost and pandas have not been added to the project:

    uv add catboost pandas

For reproducibility, use the same CatBoost version that was used
to train the model in Colab.

## Run Predictions

Keep these files together inside `src/catboost_local/`:

- `predict.py`
- `student_catboost.cbm`

Run from the project root:

    uv run python src/catboost_local/predict.py

The prediction script should:

1. Load the saved model.
2. Create or read new student records.
3. Predict the outcomes and pass probabilities.
4. Print the results.

The model does not need to be retrained for each prediction.

New records must use the same input columns and compatible data
types as the training data. Keep category spelling consistent.

## Open the HTML Report

Double-click `catboost_report.html` in File Explorer.

Alternatively, run this from the project root:

    Start-Process .\catboost_report.html

The report includes:

- A simple explanation of CatBoost
- The four model inputs
- Three example predictions
- An explanation of categorical input handling
- The training and evaluation workflow
- Project limitations

Use the report’s **Print / Save as PDF** button to export it.

The HTML report contains fixed example results. It does not run
the model or automatically update when `predict.py` runs.

## Evaluation

The notebook evaluates predictions using:

- Accuracy
- A classification report
- A confusion matrix

Measured test results have not been included here.
The example pass probabilities should not be treated as test accuracy.

## Limitations

- Training data is artificial.
- Predictions are estimates, not guaranteed outcomes.
- The project does not establish that any input causes better performance.
- No comparison against other models has been reported.
- Real student data and further validation are needed before practical use.

## Technologies

- Python
- CatBoost
- pandas and NumPy
- scikit-learn for data splitting and evaluation
- Google Colab for training
- uv for local dependency management
- HTML and CSS for reporting
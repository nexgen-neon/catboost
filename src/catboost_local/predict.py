from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


# Locate the model beside this Python script.
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "student_catboost.cbm"
)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Place student_catboost.cbm here: {MODEL_PATH}"
        )

    # Load the trained model. No fit() or retraining required.
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    new_students = pd.DataFrame({
        "study_hours": [6.5, 1.0, 4.0],
        "attendance": [92.0, 50.0, 75.0],
        "learning_method": [
            "Classroom",
            "Self Study",
            "Online",
        ],
        "internet_access": [
            "Yes",
            "No",
            "Yes",
        ],
    })

    # Preserve the feature order used during training.
    new_students = new_students[model.feature_names_]

    predictions = (
        model.predict(new_students)
        .ravel()
        .astype(int)
    )

    probabilities = model.predict_proba(new_students)

    # Locate the probability column corresponding to Pass = 1.
    pass_index = list(model.classes_).index(1)

    results = new_students.copy()

    results["prediction"] = [
        "Pass" if label == 1 else "Fail"
        for label in predictions
    ]

    results["pass_probability"] = (
        probabilities[:, pass_index].round(3)
    )

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
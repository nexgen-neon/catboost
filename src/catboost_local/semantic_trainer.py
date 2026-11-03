from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT / "data" / "semantic_labels_all_records.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "semantic_model"

TARGET_TRAIN_ROWS = 1000
RANDOM_SEED = 42


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    required_columns = [
        "source_dataset",
        "column_name",
        "target_category",
    ]

    df = pd.read_csv(
        DATA_FILE,
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = df[column].str.strip()

    df["target_category"] = df["target_category"].str.lower()

    # Unknown means unresolved, not a semantic class to learn.
    usable = (
        df["source_dataset"].ne("")
        & df["column_name"].ne("")
        & ~df["target_category"].isin(["", "unknown"])
    )

    excluded = df.loc[~usable].copy()
    df = df.loc[usable].reset_index(drop=True)

    print(f"Usable records: {len(df)}")
    print(f"Excluded records: {len(excluded)}")

    if len(df) <= TARGET_TRAIN_ROWS:
        raise ValueError(
            "More than 1,000 usable records are required "
            "for this training size and a separate test set."
        )

    return df, excluded


def split_dataset(df):
    # GroupShuffleSplit splits groups, not individual rows.
    # Try candidate splits and choose the closest to 1,000
    # training rows, without using model performance.
    splitter = GroupShuffleSplit(
        n_splits=500,
        test_size=0.22,
        random_state=RANDOM_SEED,
    )

    all_classes = set(df["target_category"])
    best_split = None
    smallest_difference = float("inf")

    for train_indices, test_indices in splitter.split(
        df,
        groups=df["source_dataset"],
    ):
        train_classes = set(
            df.iloc[train_indices]["target_category"]
        )

        # Every class must be available for the model to learn.
        if train_classes != all_classes:
            continue

        difference = abs(
            len(train_indices) - TARGET_TRAIN_ROWS
        )

        if difference < smallest_difference:
            smallest_difference = difference
            best_split = (train_indices, test_indices)

    if best_split is None:
        raise ValueError(
            "Could not find a suitable grouped split "
            "with every class represented in training."
        )

    train_indices, test_indices = best_split

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    assert set(train_df["source_dataset"]).isdisjoint(
        set(test_df["source_dataset"])
    )

    return train_df, test_df


def main():
    df, excluded = load_dataset()
    train_df, test_df = split_dataset(df)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIRECTORY / "train.csv", index=False
    )
    test_df.to_csv(
        OUTPUT_DIRECTORY / "test.csv", index=False
    )
    excluded.to_csv(
        OUTPUT_DIRECTORY / "excluded.csv", index=False
    )

    print(f"\nTraining records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")

    distribution = pd.concat(
        [
            train_df["target_category"]
            .value_counts()
            .rename("train"),
            test_df["target_category"]
            .value_counts()
            .rename("test"),
        ],
        axis=1,
    ).fillna(0).astype(int)

    print("\nClass distribution:")
    print(distribution.to_string())

    # Convert short character patterns into numeric features.
    # For example, "user_id" and "product_id" share "_id".
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        binary=True,
        max_features=5000,
    )

    # Learn the vocabulary ONLY from the training names.
    X_train = vectorizer.fit_transform(
        train_df["column_name"]
    ).astype("float32")

    X_test = vectorizer.transform(
        test_df["column_name"]
    ).astype("float32")

    y_train = train_df["target_category"]
    y_test = test_df["target_category"]

    model = CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=RANDOM_SEED,
        verbose=100,
        allow_writing_files=False,
        thread_count=4,
    )

    # The test set is not used to tune or stop training.
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).ravel()
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=list(model.classes_),
        zero_division=0,
    )

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)

    results = test_df.copy()
    results["predicted_category"] = predictions
    results["model_probability"] = probabilities.max(axis=1)
    results["correct"] = (
        results["target_category"]
        == results["predicted_category"]
    )

    results.to_csv(
        OUTPUT_DIRECTORY / "test_predictions.csv",
        index=False,
    )

    classes = list(model.classes_)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes,
    )

    pd.DataFrame(
        matrix,
        index=pd.Index(classes, name="actual"),
        columns=classes,
    ).to_csv(
        OUTPUT_DIRECTORY / "confusion_matrix.csv"
    )

    (OUTPUT_DIRECTORY / "evaluation.txt").write_text(
        f"Training rows: {len(train_df)}\n"
        f"Test rows: {len(test_df)}\n"
        f"Accuracy: {accuracy:.2%}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    model.save_model(
        str(OUTPUT_DIRECTORY / "semantic_catboost.cbm")
    )

    joblib.dump(
        vectorizer,
        OUTPUT_DIRECTORY / "column_name_vectorizer.joblib",
    )

    print(f"\nSaved results and model to:\n{OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
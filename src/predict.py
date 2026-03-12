import argparse
import csv
import os
import pickle
import sys

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None
try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


def parse_features(csv_features):
    try:
        return [float(value) for value in next(csv.reader([csv_features]))]
    except (ValueError, csv.Error) as exc:
        raise ValueError("Features must be numeric CSV values") from exc


def load_model(model_path):
    if model_path is None:
        return None
    if joblib is not None:
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    with open(model_path, "rb") as file_handle:
        return pickle.load(file_handle)


def main():
    feature_names = [
        "years_experience",
        "skills_match_score",
        "project_count",
        "resume_length",
        "github_activity",
    ]
    education_options = [
        "High School",
        "Masters",
        "PhD",
        "Bachelors",
    ]
    parser = argparse.ArgumentParser(description="Run a pickle model prediction")
    parser.add_argument("--model", help="Path to .pkl model file")
    parser.add_argument(
        "--features",
        help="Comma-separated numeric feature list",
    )
    args = parser.parse_args()

    model_path = args.model
    model_label = ""
    model_accuracy = None
    if not model_path:
        print("Select model:")
        print("1) Logistic Regression")
        print("2) Random Forest")
        choice = input("Enter choice (1/2): ").strip()
        if choice == "1":
            model_path = r"C:\Users\User\Downloads\IT20\logistic_regression_pipeline.pkl"
            model_label = "Logistic Regression"
            model_accuracy = 0.9063
        elif choice == "2":
            model_path = r"C:\Users\User\Downloads\IT20\random_forest_pipeline.pkl"
            model_label = "Random Forest"
            model_accuracy = 0.901
        else:
            print("Invalid choice.", file=sys.stderr)
            return 2
    education_value = "Bachelors"

    if args.features:
        try:
            features = parse_features(args.features)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if len(features) == 8:
            if features[5] == 1:
                education_value = "High School"
            elif features[6] == 1:
                education_value = "Masters"
            elif features[7] == 1:
                education_value = "PhD"
    else:
        features = []
        for name in feature_names:
            while True:
                raw_value = input(f"Enter {name}: ").strip()
                try:
                    features.append(float(raw_value))
                except ValueError:
                    print("Please enter a numeric value.", file=sys.stderr)
                    continue
                break

        print("Select education level:")
        for idx, label in enumerate(education_options, start=1):
            print(f"{idx}) {label}")
        choice = input("Enter choice (1-4): ").strip()
        if choice not in {"1", "2", "3", "4"}:
            print("Invalid choice.", file=sys.stderr)
            return 2
        education_one_hot = [0.0, 0.0, 0.0]
        education_value = "Bachelors"
        if choice in {"1", "2", "3"}:
            education_one_hot[int(choice) - 1] = 1.0
            education_value = education_options[int(choice) - 1]
        features.extend(education_one_hot)

    model = load_model(model_path)
    if model is None:
        print("Model could not be loaded.", file=sys.stderr)
        return 3

    if pd is None:
        print("pandas is required for pipeline models.", file=sys.stderr)
        return 3

    try:
        model_input = pd.DataFrame(
            [
                {
                    "years_experience": features[0],
                    "skills_match_score": features[1],
                    "project_count": features[2],
                    "resume_length": features[3],
                    "github_activity": features[4],
                    "education_level": education_value,
                }
            ]
        )
        prediction = model.predict(model_input)
    except Exception as exc:  # noqa: BLE001
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 3

    prediction_raw = prediction[0]

    probability = None
    probability_label = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(model_input)
            class_indices = {label: idx for idx, label in enumerate(model.classes_)}
            predicted_class = prediction_raw
            if predicted_class in class_indices:
                probability = float(proba[0][class_indices[predicted_class]])
                probability_label = "Predicted class"
            elif 1 in class_indices:
                probability = float(proba[0][class_indices[1]])
                probability_label = "Shortlisted"
            elif "Yes" in class_indices:
                probability = float(proba[0][class_indices["Yes"]])
                probability_label = "Shortlisted"
        except Exception:  # noqa: BLE001
            probability = None

    label_map = {
        0: "Not Shortlisted",
        1: "Shortlisted",
        "No": "Not Shortlisted",
        "Yes": "Shortlisted",
    }
    prediction_raw = prediction[0]
    if isinstance(prediction_raw, str):
        prediction_label = label_map.get(prediction_raw, prediction_raw)
    else:
        prediction_label = label_map.get(int(prediction_raw), str(prediction_raw))

    if model_label:
        print(f"Model: {model_label}")
    if model_accuracy is not None:
        print(f"Training accuracy: {model_accuracy:.4f}")
    print(f"Prediction: {prediction_label}")
    if probability is not None:
        label_suffix = f" ({probability_label})" if probability_label else ""
        print(f"Probability{label_suffix}: {probability * 100:.2f}%")
    else:
        print("Probability: unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

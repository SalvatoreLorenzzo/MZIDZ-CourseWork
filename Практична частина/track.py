import argparse
import json
from datetime import date, timedelta

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from models import RESULTS, TEST_START, VALIDATION_START, read_data, run_experiment

MODEL_NAME = "bike_rentals_automl"


def connect_tracking():
    RESULTS.mkdir(exist_ok=True)
    path = (RESULTS / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{path}")
    mlflow.set_registry_uri(f"sqlite:///{path}")


def run_and_log():
    connect_tracking()
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("Велопрокат")
    if experiment is None:
        experiment_id = client.create_experiment(
            "Велопрокат", artifact_location=(RESULTS / "artifacts").as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    test, predictions, details, models = run_experiment()
    predictions.to_csv(RESULTS / "predictions.csv", index_label="date")
    selected = min(models, key=lambda name: details[name]["validation_mae"])

    for name, result in details.items():
        with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
            mlflow.log_params({
                "validation_start": VALIDATION_START, "test_start": TEST_START,
                "test_end": test.index[-1].date().isoformat(), "horizon_days": 1,
            })
            for key, value in result.items():
                if isinstance(value, float):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, value)
            mlflow.log_dict(
                {day.date().isoformat(): row.to_dict()
                 for day, row in predictions[["actual", name]].iterrows()},
                "predictions.json",
            )
            if name == selected:
                columns = details[selected]["features"]
                inputs = test[columns]
                mlflow.sklearn.log_model(
                    models[selected], name="model", registered_model_name=MODEL_NAME,
                    serialization_format="cloudpickle",
                    signature=infer_signature(inputs.astype("float64"), predictions[selected]),
                    input_example=inputs.iloc[:2].astype("float64"),
                    metadata={"features": columns},
                )
            result["run_id"] = run.info.run_id

    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    version = next(item for item in versions if item.run_id == details[selected]["run_id"])
    model_uri = f"models:/{MODEL_NAME}/{version.version}"
    client.set_registered_model_alias(MODEL_NAME, "validation_best", version.version)
    loaded = mlflow.sklearn.load_model(model_uri)
    reproduced = loaded.predict(test[details[selected]["features"]])
    np.testing.assert_allclose(reproduced, predictions[selected].to_numpy())
    report = {"comparison": details, "registered": selected, "model_uri": model_uri}
    (RESULTS / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def predict_next_day(day, holiday):
    connect_tracking()
    data = read_data()
    last_day = data.index[-1].date()
    if day != last_day + timedelta(days=1):
        raise SystemExit(f"Наступна дата після історії в сховищі — {last_day + timedelta(days=1)}.")

    totals = data["total"]
    features = pd.DataFrame([{
        "lag_1": totals.iloc[-1], "lag_7": totals.iloc[-7], "lag_14": totals.iloc[-14],
        "mean_7": totals.iloc[-7:].mean(), "month": day.month, "weekday": day.weekday(),
        "holiday": int(holiday), "working_day": int(day.weekday() < 5 and not holiday),
    }])
    uri = f"models:/{MODEL_NAME}@validation_best"
    columns = mlflow.models.get_model_info(uri).metadata["features"]
    model = mlflow.sklearn.load_model(uri)
    prediction = float(model.predict(features[columns])[0])
    print(f"{day}: прогноз {prediction:.0f} прокатів")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=date.fromisoformat)
    parser.add_argument("--holiday", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.predict:
        if args.holiday is None:
            parser.error("Для прогнозу вкажіть --holiday 0 або --holiday 1.")
        predict_next_day(args.predict, bool(args.holiday))
    else:
        run_and_log()


if __name__ == "__main__":
    main()

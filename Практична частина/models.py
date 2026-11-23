import json
import sqlite3
from pathlib import Path
from time import perf_counter
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from etl import DATABASE

RESULTS = Path(__file__).resolve().parent / "results"
VALIDATION_START = "2012-07-01"
TEST_START = "2012-10-01"
TARGET = "total"
HISTORY = ["lag_1", "lag_7", "lag_14", "mean_7"]
CALENDAR = ["month", "weekday", "holiday", "working_day"]
LEARNERS = ["rf", "extra_tree", "lgbm"]


def read_data():
    with sqlite3.connect(DATABASE) as connection:
        frame = pd.read_sql_query(
            """SELECT day, total, month, weekday, holiday, working_day,
                      lag_1, lag_7, lag_14, mean_7
               FROM daily_rentals JOIN calendar_day USING (day)
               LEFT JOIN forecast_features USING (day)
               ORDER BY day""",
            connection, parse_dates=["day"], index_col="day",
        )
    return frame.asfreq("D")


def score(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
    }


def sarima_fit(series, order, seasonal_order):
    with catch_warnings():
        simplefilter("ignore", ConvergenceWarning)
        return SARIMAX(series, order=order, seasonal_order=seasonal_order).fit(
            disp=False, maxiter=150,
        )


def walk_sarima(fitted, actual):
    predictions = []
    for i in range(len(actual)):
        predictions.append(max(0.0, float(fitted.forecast().iloc[0])))
        fitted = fitted.extend(actual.iloc[i:i + 1])
    return np.array(predictions)


def sarima_forecast(train, validation, full_train, test):
    start = perf_counter()
    candidates = [
        ((1, 0, 0), (1, 0, 0, 7)),
        ((1, 1, 0), (1, 0, 0, 7)),
        ((0, 1, 1), (0, 1, 1, 7)),
    ]
    scores = {}
    for order, seasonal_order in candidates:
        fitted = sarima_fit(train[TARGET], order, seasonal_order)
        if fitted.mle_retvals["converged"]:
            scores[order, seasonal_order] = score(
                validation[TARGET], walk_sarima(fitted, validation[TARGET]),
            )["mae"]
    if not scores:
        raise RuntimeError("Жодна конфігурація SARIMA не зійшлася.")
    selected = min(scores, key=scores.get)
    search_seconds = perf_counter() - start

    start = perf_counter()
    fitted = sarima_fit(full_train[TARGET], *selected)
    if not fitted.mle_retvals["converged"]:
        raise RuntimeError(f"Повторне навчання SARIMA {selected} не зійшлося.")
    fit_seconds = perf_counter() - start
    predicted = walk_sarima(fitted, test[TARGET])
    return predicted, {
        "validation_mae": scores[selected],
        "search_seconds": search_seconds, "fit_seconds": fit_seconds,
        "order": str(selected[0]), "seasonal_order": str(selected[1]),
    }


def automl_forecast(train, validation, full_train, test, columns):
    start = perf_counter()
    automl = AutoML()
    automl.fit(
        X_train=train[columns], y_train=train[TARGET],
        X_val=validation[columns], y_val=validation[TARGET],
        task="regression", metric="mae", eval_method="holdout", retrain_full=False,
        estimator_list=LEARNERS, max_iter=30, time_budget=25, sample=False,
        seed=42, n_jobs=1, verbose=0, mlflow_logging=False,
    )
    search_seconds = perf_counter() - start
    model = clone(automl.model.estimator)
    start = perf_counter()
    model.fit(full_train[columns], full_train[TARGET])
    fit_seconds = perf_counter() - start
    predicted = model.predict(test[columns])
    return predicted, {
        "validation_mae": float(automl.best_loss),
        "search_seconds": search_seconds, "fit_seconds": fit_seconds,
        "estimator": automl.best_estimator, "config": automl.best_config,
        "features": columns, "seed": 42, "max_iterations": 30, "budget_seconds": 25,
    }, model


def run_experiment():
    data = read_data()
    train = data.loc[data.index < VALIDATION_START]
    validation = data.loc[(data.index >= VALIDATION_START) & (data.index < TEST_START)]
    full_train = data.loc[data.index < TEST_START]
    test = data.loc[data.index >= TEST_START]
    train_features = train.dropna(subset=HISTORY)
    full_features = full_train.dropna(subset=HISTORY)

    predictions = pd.DataFrame({"actual": test[TARGET], "seasonal_naive": test["lag_7"]})
    details = {"seasonal_naive": {
        "validation_mae": score(validation[TARGET], validation["lag_7"])["mae"],
    }}
    predictions["sarima"], details["sarima"] = sarima_forecast(
        train, validation, full_train, test,
    )
    models = {}
    for name, columns in (("automl_history", HISTORY), ("automl_calendar", HISTORY + CALENDAR)):
        predictions[name], details[name], models[name] = automl_forecast(
            train_features, validation, full_features, test, columns,
        )
    for name, result in details.items():
        result.update(score(predictions["actual"], predictions[name]))

    return test, predictions, details, models


def main():
    _, predictions, details, _ = run_experiment()
    RESULTS.mkdir(exist_ok=True)
    predictions.to_csv(RESULTS / "predictions.csv", index_label="date")
    print(json.dumps(details, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

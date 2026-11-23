import csv
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT.parent / "Bike Sharing" / "day.csv"
DATABASE = PROJECT / "велопрокат.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_day (
    day TEXT PRIMARY KEY,
    month INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    weekday INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    holiday INTEGER NOT NULL CHECK (holiday IN (0, 1)),
    working_day INTEGER NOT NULL CHECK (working_day IN (0, 1))
);
CREATE TABLE IF NOT EXISTS daily_rentals (
    day TEXT PRIMARY KEY REFERENCES calendar_day(day),
    source_id INTEGER NOT NULL UNIQUE,
    casual INTEGER NOT NULL CHECK (casual >= 0),
    registered INTEGER NOT NULL CHECK (registered >= 0),
    total INTEGER NOT NULL CHECK (total = casual + registered),
    weather_code INTEGER NOT NULL CHECK (weather_code BETWEEN 1 AND 4),
    temperature REAL NOT NULL,
    perceived_temperature REAL NOT NULL,
    humidity REAL NOT NULL,
    wind_speed REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_features (
    day TEXT PRIMARY KEY REFERENCES calendar_day(day),
    lag_1 INTEGER NOT NULL,
    lag_7 INTEGER NOT NULL,
    lag_14 INTEGER NOT NULL,
    mean_7 REAL NOT NULL
);
"""


def read_source():
    with SOURCE.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    if len(rows) < 15:
        raise ValueError("Для формування ознак потрібно щонайменше 15 днів.")

    calendar, rentals = [], []
    source_ids = set()
    previous_day = None
    for row in rows:
        if None in row or any(not value or not value.strip() for value in row.values()):
            raise ValueError(f"Неповний запис CSV: {row}")

        day = date.fromisoformat(row["dteday"])
        source_id = int(row["instant"])
        casual = int(row["casual"])
        registered = int(row["registered"])
        total = int(row["cnt"])

        if previous_day is not None and day != previous_day + timedelta(days=1):
            raise ValueError(f"Порушено послідовність дат: {previous_day}, {day}")
        if source_id in source_ids:
            raise ValueError(f"Повторний ідентифікатор: {source_id}")
        if min(casual, registered, total) < 0 or total != casual + registered:
            raise ValueError(f"Помилка в кількості прокатів за {day}.")

        source_ids.add(source_id)
        previous_day = day
        calendar.append((
            day.isoformat(), day.month, day.weekday(),
            int(row["holiday"]), int(row["workingday"]),
        ))
        rentals.append((
            day.isoformat(), source_id, casual, registered, total,
            int(row["weathersit"]), float(row["temp"]), float(row["atemp"]),
            float(row["hum"]), float(row["windspeed"]),
        ))
    return calendar, rentals


def build_features(rentals):
    totals = [record[4] for record in rentals]
    return [
        (rentals[i][0], totals[i - 1], totals[i - 7], totals[i - 14], mean(totals[i - 7:i]))
        for i in range(14, len(rentals))
    ]


def main():
    calendar, rentals = read_source()
    features = build_features(rentals)

    with sqlite3.connect(DATABASE) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        connection.execute("BEGIN")
        connection.execute("DELETE FROM forecast_features")
        connection.execute("DELETE FROM daily_rentals")
        connection.execute("DELETE FROM calendar_day")
        connection.executemany("INSERT INTO calendar_day VALUES (?, ?, ?, ?, ?)", calendar)
        connection.executemany(
            "INSERT INTO daily_rentals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rentals
        )
        connection.executemany("INSERT INTO forecast_features VALUES (?, ?, ?, ?, ?)", features)
        days, minimum, maximum, average = connection.execute(
            "SELECT COUNT(*), MIN(total), MAX(total), AVG(total) FROM daily_rentals"
        ).fetchone()
        feature_days = connection.execute("SELECT COUNT(*) FROM forecast_features").fetchone()[0]
        weekdays = connection.execute(
            """SELECT weekday, AVG(total)
               FROM daily_rentals JOIN calendar_day USING (day)
               GROUP BY weekday ORDER BY weekday"""
        ).fetchall()

    print("Завантаження добових даних про велопрокат")
    print(f"Період: {calendar[0][0]} — {calendar[-1][0]}")
    print(f"Днів у сховищі: {days}, днів з повною історією ознак: {feature_days}")
    print(f"Прокатів за добу: мінімум {minimum}, максимум {maximum}, середнє {average:.2f}")
    print("Перевірки дат, дублікатів і контрольних сум пройдено.")
    print("Середня кількість прокатів за днями тижня:")
    for weekday, average in weekdays:
        print(f"  {('Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд')[weekday]}: {average:.2f}")
    print(f"Файл сховища: {DATABASE.name}")


if __name__ == "__main__":
    main()

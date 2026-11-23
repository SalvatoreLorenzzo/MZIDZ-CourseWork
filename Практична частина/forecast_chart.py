import csv
import json
import tkinter as tk
from datetime import date
from pathlib import Path

from charts import grid

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    report = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    selected = report["registered"]
    with (RESULTS / "predictions.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    root = tk.Tk()
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=930, height=365, bg="white", highlightthickness=0)
    canvas.pack()

    left, right, top, bottom = 60, 900, 48, 310
    series = [
        ("actual", "Фактичні значення", "#263238"),
        ("sarima", "SARIMA", "#957343"),
        (selected, "AutoML", "#2371ad"),
    ]
    highest = max(float(row[key]) for row in rows for key, _, _ in series)
    maximum = 2000 * (int(highest / 2000) + 1)
    grid(canvas, left, top, right - left, bottom - top, range(0, maximum + 1, 2000))

    for index, (key, label, color) in enumerate(series):
        points = []
        for i, row in enumerate(rows):
            x = left + (right - left) * i / (len(rows) - 1)
            y = bottom - (bottom - top) * float(row[key]) / maximum
            points.extend((x, y))
        canvas.create_line(*points, fill=color, width=2)
        legend_x = left + index * 175
        canvas.create_line(legend_x, 22, legend_x + 25, 22, fill=color, width=2)
        canvas.create_text(legend_x + 32, 22, text=label, anchor="w", font=("Segoe UI", 10))

    ticks = {"2012-10-01", "2012-10-15", "2012-11-01", "2012-11-15",
             "2012-12-01", "2012-12-15", "2012-12-31"}
    for i, row in enumerate(rows):
        if row["date"] in ticks:
            x = left + (right - left) * i / (len(rows) - 1)
            canvas.create_text(x, bottom + 18, text=date.fromisoformat(row["date"]).strftime("%d.%m"),
                               font=("Segoe UI", 10))
    root.mainloop()


if __name__ == "__main__":
    main()

import sqlite3
import tkinter as tk
from datetime import date

from etl import DATABASE


def grid(canvas, x, y, width, height, ticks):
    for value in range(0, ticks[-1] + 1, ticks[1]):
        line_y = y + height * (1 - value / ticks[-1])
        canvas.create_line(x, line_y, x + width, line_y, fill="#d0d7de")
        canvas.create_text(x - 9, line_y, text=str(value), anchor="e", font=("Segoe UI", 10))


def main():
    if not DATABASE.is_file():
        raise SystemExit("Спочатку виконайте python etl.py.")

    with sqlite3.connect(DATABASE) as connection:
        series = connection.execute("SELECT day, total FROM daily_rentals ORDER BY day").fetchall()
        weekday = connection.execute(
            """SELECT weekday, AVG(total)
               FROM daily_rentals JOIN calendar_day USING (day)
               GROUP BY weekday ORDER BY weekday"""
        ).fetchall()

    root = tk.Tk()
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=950, height=575, bg="white", highlightthickness=0)
    canvas.pack()

    start, end = 64, 915
    top, bottom = 62, 300
    canvas.create_text(start, 25, text="Кількість прокатів за добу", anchor="w",
                       font=("Segoe UI", 14, "bold"))
    grid(canvas, start, top, end - start, bottom - top, range(0, 10001, 2000))

    points = []
    for i, (_, total) in enumerate(series):
        points.extend((start + (end - start) * i / (len(series) - 1),
                       bottom - (bottom - top) * total / 10000))
    canvas.create_line(*points, fill="#2369a3")

    tick_dates = {"2011-01-01", "2011-07-01", "2012-01-01", "2012-07-01", "2012-12-31"}
    for i, (day, _) in enumerate(series):
        if day in tick_dates:
            x = start + (end - start) * i / (len(series) - 1)
            canvas.create_text(x, bottom + 15, text=date.fromisoformat(day).strftime("%m.%Y"),
                               font=("Segoe UI", 10))

    top, bottom = 383, 527
    canvas.create_text(start, 353, text="Середня кількість прокатів за днем тижня",
                       anchor="w", font=("Segoe UI", 14, "bold"))
    grid(canvas, start, top, end - start, bottom - top, range(0, 6000, 1000))
    spacing = (end - start) / 7
    for day, avg in weekday:
        x = start + spacing * (day + 0.5)
        y = bottom - (bottom - top) * avg / 5000
        canvas.create_rectangle(x - 34, y, x + 34, bottom, fill="#2369a3", outline="")
        canvas.create_text(x, y - 11, text=f"{avg:.0f}", font=("Segoe UI", 10))
        canvas.create_text(x, bottom + 15, text=("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд")[day],
                           font=("Segoe UI", 10))

    root.mainloop()


if __name__ == "__main__":
    main()

# Прогнозування добової кількості велопрокатів

Код практичної частини курсової роботи. Конвеєр завантажує добові дані Capital Bikeshare у SQLite, порівнює базовий прогноз, SARIMA та два варіанти FLAML і реєструє обрану модель AutoML у MLflow.

## Запуск

Команди для PowerShell із Python 3.13. Перед завантаженням даних створюється віртуальне середовище та встановлюються залежності.

```powershell
Set-Location -LiteralPath 'Практична частина'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe etl.py
.\.venv\Scripts\python.exe track.py
```

Прогнози, метрики та база MLflow зберігаються в папці `results` поруч зі скриптами. Графік тестових прогнозів відкривається командою `.\.venv\Scripts\python.exe forecast_chart.py` у середовищі з Tkinter.

Команда нижче формує демонстраційний прогноз на наступну добу після кінця набору.

```powershell
.\.venv\Scripts\python.exe track.py --predict 2013-01-01 --holiday 1
```

## Дані

Використано `day.csv` без змін із набору [Bike Sharing](https://doi.org/10.24432/C5W894), Hadi Fanaee-T, UCI Machine Learning Repository, 2013. Ліцензія набору — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Супровідна публікація — [Event labeling combining ensemble detectors and background knowledge](https://doi.org/10.1007/s13748-013-0040-3).

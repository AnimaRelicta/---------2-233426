Для запуска проекта необходимо

```bash
# синхронизация проекта и сопоставление зависимостей
uv sync

# пакеты и доступ к MOEX и Hugging Face
uv run task2_sync.py check
uv run task2_sync.py candles --start 2019-09-01 --end 2024-07-10

# раздел «Экономика» из Lenta-news, период 2019-12-15..2024-06-30
uv run task2_sync.py news

# образцы, графики, отчёт
uv run task2_sync.py build --out out
```

Создайте на Drive папку `MyDrive/lab2` и положите в неё:

| Что |
|---|
| `out/` целиком (`samples.csv`, `img/`, `exclusions.csv`, `bad_images.csv`, `data_report.md`) |
| `data/candles.csv` (положите как `lab2/data/candles.csv`) |

Дальше работа идет в юпитер ноутбуке, так как локальных мощностей нет для запуска скриптов,
Используем ноутбук `task3_7.ipynb`, подключаем наш гугл диск и запускаем все ячейки


Ссылка на гугл диск

https://drive.google.com/drive/folders/1x0XgkiPujKyOfmUnYdLWOB3REupfEsSr?usp=sharing
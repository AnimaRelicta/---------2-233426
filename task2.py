#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа 2, задание 2. Синхронизация данных: новость x тикер x окно котировок x свечное изображение.
ВСЯ логика задания в одном файле. Постановка зафиксирована:
   * новости: ТОЛЬКО Lenta.ru, набор Hugging Face rustemgareev/lenta-news, раздел (topic) «Экономика»,
     период 2019-12-15 .. 2024-06-30 (время в наборе без пояса, принято МСК);
   * тикер: по умолчанию SBER (Сбербанк), другой через --ticker (задание 4); MOEX, режим TQBR, дневные свечи;
   * цель: направление доходности следующего торгового дня, окно 20 дневных свечей, свечное изображение.

===================================== ИНСТРУКЦИЯ ПО ЗАПУСКУ =====================================
0. Python 3.9+ и пакеты:
       pip install requests pandas numpy matplotlib pillow tabulate pyarrow
   Запускать все команды из одной папки (рядом создадутся data/ и out/).

1. Проверка окружения и доступа (всегда начинайте с неё):
       python task2_sync.py check
   Покажет OK/FAIL для MOEX ISS и Hugging Face. При FAIL читайте причину (нет интернета, нужен VPN/прокси).

2. Свечи SBER с MOEX -> data/candles.csv. Период берётся с запасом: 20 свечей ДО первой новости и
   несколько дней ПОСЛЕ последней (нужен следующий торговый день):
       python task2_sync.py candles
   (по умолчанию --start 2019-09-01 --end 2024-07-10)

3. Новости Lenta, раздел «Экономика» -> data/news.csv:
       python task2_sync.py news
   Скачивает data/lenta_train.parquet (один раз; файл большой, до ~1 ГБ) и отбирает topic == «Экономика».
   Если автоскачивание не работает, скачайте train.parquet вручную со страницы набора
   https://huggingface.co/datasets/rustemgareev/lenta-news (Files -> data/train.parquet) и укажите путь:
       python task2_sync.py news --file путь/к/train.parquet
   Эквивалент в ноутбуке: df = pd.read_parquet("hf://datasets/rustemgareev/lenta-news/data/train.parquet");
   df[df.topic == "Экономика"]   (скрипт делает то же самое, но читает только нужные колонки).

4. Сборка датасета -> out/:
       python task2_sync.py build
   Параметры: --drop-intraday (убрать новости, опубликованные во время торговой сессии: по дневным свечам
   неясно, какая часть дневного движения случилась ДО новости), --out папка.
   Если образцов 0, скрипт не падает, а печатает причины исключения.

5. Проверка кода без интернета, на синтетике (цифры в отчёт не переносить):
       python task2_sync.py demo           # результат в out_demo/

Результаты build (папка out/):
   samples.csv       образец = новость x тикер: sample_id, ticker, source, published, session, anchor_date,
                     target_date, text, n_tickers, ret_next, label, volume_zero_in_window, image, split
   img/              PNG 224x224, по одному файлу на якорную дату (образцы с одним якорем делят картинку)
   exclusions.csv    журнал исключённых наблюдений с причиной
   bad_images.csv    битые / пустые изображения
   data_report.md    описательная статистика для разделов 3 и 6 отчёта
Окно OHLCV воспроизводится по (ticker, anchor_date) из data/candles.csv.

Правила синхронизации: все времена в MSK; якорь A = последний торговый день, сессия которого закрыта к
моменту публикации; окно = 20 свечей до A включительно; цель = знак close(A+1)/close(A)-1; календарь
торговых дней строится по свечам; пропуски не интерполируются, образец исключается.
Ограничения набора: лицензия Lenta-news на странице «unknown» (корпус не публиковать, только ссылки);
Lenta - общий источник, новостей именно про Сбербанк в разделе «Экономика» немного, смотрите data_report.md.
=================================================================================================
"""
import argparse
import bisect
import hashlib
import html
import os
import re
import sys
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


# ============================================ КОНФИГУРАЦИЯ ============================================
MSK = ZoneInfo("Europe/Moscow")
ISS = os.environ.get("ISS_URL", "https://iss.moex.com/iss")
BOARD = "TQBR"
# Основная сессия акций MOEX: проверьте по регламенту на дату выборки и при необходимости поправьте.
OPEN_HHMM = (9, 50)
CLOSE_HHMM = (18, 50)
WINDOW, HORIZON, EMBARGO_DAYS = 20, 1, 1
SPLIT = (0.70, 0.15, 0.15)
MAX_TICKERS_PER_NEWS = 3
MAX_GAP_DAYS = 10   # якорь -> цель дальше 10 календарных дней = остановка рынка (напр. 28.02-23.03.2022): образец исключается
NEAR_DUP_JACCARD, NEAR_DUP_HOURS = 0.8, 48
NAIVE_TZ = MSK  # время в Lenta-news без пояса: принято МСК (все такие записи помечаются naive_tz и считаются в отчёте)

# Единственный тикер. Алиасы: regex без учёта регистра.
TICKERS = {"SBER": [r"сбербанк\w*", r"\bсбер(а|у|ом|е)?\b", r"\bSBER\b"]}
# Единственный источник новостей и раздел.
LENTA_URL = "https://huggingface.co/datasets/rustemgareev/lenta-news/resolve/main/data/train.parquet"
TOPIC = "Экономика"
NEWS_START, NEWS_END = "2019-12-15", "2024-06-30"
CANDLES_START, CANDLES_END = "2019-09-01", "2024-07-10"    # запас на окно в 20 свечей и на A+1
DATA_DIR = "data"
CANDLES_CSV = os.path.join(DATA_DIR, "candles.csv")
NEWS_CSV = os.path.join(DATA_DIR, "news.csv")
LENTA_PARQUET = os.path.join(DATA_DIR, "lenta_train.parquet")
HEADERS = {"User-Agent": "Mozilla/5.0 (lab2-research)"}
_PATTERNS = {t: [re.compile(p, re.I) for p in ps] for t, ps in TICKERS.items()}

# Заготовки алиасов для других тикеров (задание 4 лабораторной: вторая компания). Свои задаются флагом --aliases.
PRESETS = {
    "GAZP": [r"\bгазпром(а|у|ом|е)?\b(?!\s+нефт|-\w)", r"\bGAZP\b"],      # без «Газпром нефть» и «Газпром-Медиа»
    "LKOH": [r"\bлукойл\w*", r"\bLKOH\b"],
    "ROSN": [r"\bроснефт\w*", r"\bROSN\b"],
    "GMKN": [r"\bнорникел\w*", r"\bнорильск\w*\s+никел\w*", r"\bGMKN\b"],
    "VTBR": [r"\bвтб\b", r"\bVTBR\b"],
}


def configure(ticker=None, aliases=None, data_dir=None, news_csv=None):
    """Переключает скрипт на другой тикер без правки кода. Без аргументов всё как раньше: SBER и папка data/.
    data_dir: куда писать свечи этого тикера (не затирает data/candles.csv от SBER).
    news_csv: откуда читать новости; news.csv от SBER подходит как есть: в нём весь раздел без фильтра по тикеру."""
    global TICKERS, _PATTERNS, DATA_DIR, CANDLES_CSV, NEWS_CSV, LENTA_PARQUET
    if data_dir:
        DATA_DIR = data_dir
        CANDLES_CSV = os.path.join(DATA_DIR, "candles.csv")
        LENTA_PARQUET = os.path.join(DATA_DIR, "lenta_train.parquet")
        NEWS_CSV = os.path.join(DATA_DIR, "news.csv")
    if news_csv:
        NEWS_CSV = news_csv
    if ticker:
        als = [x.strip() for x in aliases.split(";") if x.strip()] if aliases else PRESETS.get(ticker)
        if not als:
            sys.exit(f"Для {ticker} нет заготовки алиасов: задайте --aliases 'regex1;regex2'")
        TICKERS = {ticker: als}
    _PATTERNS = {t: [re.compile(p, re.I) for p in ps] for t, ps in TICKERS.items()}


# ============================================ ЗАГРУЗКА ДАННЫХ ============================================
def http_get(url, **kw):
    """GET с тремя повторами; при неудаче бросает исключение с понятным текстом."""
    last = None
    for k in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url} недоступен: {type(last).__name__}: {last}")


def fetch_candles(sec, start, end, interval=24):
    """Свечи MOEX ISS (interval=24 дневные). Постранично (500 строк), с защитой от зацикливания."""
    url = f"{ISS}/engines/stock/markets/shares/boards/{BOARD}/securities/{sec}/candles.json"
    rows, cols, offset, prev_first = [], None, 0, None
    while True:
        blk = http_get(url, params={"from": start, "till": end, "interval": interval,
                                    "start": offset, "iss.meta": "off"}).json()["candles"]
        cols, data = blk["columns"], blk["data"]
        if not data or data[0] == prev_first:       # конец данных или сервер игнорирует `start`
            break
        prev_first = data[0]
        rows += data
        offset += len(data)
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    begin = pd.to_datetime(df["begin"]).dt.tz_localize(MSK)
    df["date"] = begin.dt.strftime("%Y-%m-%d")
    df["ticker"] = sec
    keep = ["ticker", "date", "open", "high", "low", "close", "volume", "value"]
    return df[[c for c in keep if c in df.columns]].sort_values("date").reset_index(drop=True)


def load_candles():
    if not os.path.exists(CANDLES_CSV):
        sys.exit(f"Нет {CANDLES_CSV}. Сначала: python task2_sync.py candles --start ГГГГ-ММ-ДД --end ГГГГ-ММ-ДД")
    df = pd.read_csv(CANDLES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _clean(t):
    t = html.unescape(re.sub(r"<[^>]+>", " ", t if isinstance(t, str) else ""))
    return re.sub(r"\s+", " ", t).strip()


def save_news(df):
    """data/news.csv; published хранится как ISO со смещением МСК."""
    os.makedirs(DATA_DIR, exist_ok=True)
    out = df.copy()
    out["published"] = out["published"].map(lambda t: t.isoformat() if pd.notna(t) else "")
    out.to_csv(NEWS_CSV, index=False)


def load_news():
    if not os.path.exists(NEWS_CSV):
        sys.exit(f"Нет {NEWS_CSV}. Сначала: python task2_sync.py news")
    df = pd.read_csv(NEWS_CSV, keep_default_na=False)
    df["published"] = pd.to_datetime(df["published"].replace("", None), utc=True, errors="coerce").dt.tz_convert(MSK)
    df["naive_tz"] = df["naive_tz"].astype(str).str.lower().eq("true")
    return df


def download(url, path):
    """Потоковое скачивание большого файла с индикатором прогресса."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); done += len(chunk)
                print(f"\rскачано {done / 1e6:.0f} МБ" + (f" из {total / 1e6:.0f}" if total else ""), end="", flush=True)
    os.replace(tmp, path)
    print()


def read_lenta(path, topic=TOPIC):
    """Читает Lenta-news (parquet или csv), оставляет раздел topic и период NEWS_START..NEWS_END."""
    cols = ["url", "date", "title", "topic", "news"]
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path, usecols=lambda c: c in cols, dtype=str, keep_default_na=False)
    else:
        df = pd.read_parquet(path, columns=cols)                   # нужен pyarrow
    total = len(df)
    df = df[df["topic"] == topic].copy()
    ts = pd.to_datetime(df["date"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:                         # в наборе время без пояса -> МСК
        naive = ts.notna()
        ts = ts.dt.tz_localize(NAIVE_TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        naive = pd.Series(False, index=ts.index)
    out = pd.DataFrame({"source": "lenta", "title": df["title"].map(_clean), "text": df["news"].map(_clean),
                        "url": df["url"].astype(str), "published": ts.dt.tz_convert(MSK), "naive_tz": naive})
    print(f"в наборе {total} новостей; раздел «{topic}»: {len(out)}")
    return out


def print_news_span(news, label="новости"):
    p = news["published"].dropna()
    if p.empty:
        print(f"[warn] {label}: ни одной распознанной даты"); return
    print(f"{label}: {len(news)} записей, период {p.min():%Y-%m-%d} .. {p.max():%Y-%m-%d}")
    if os.path.exists(CANDLES_CSV):
        c = load_candles()["date"]
        lo, hi = min(c), max(c)
        inside = ((p.dt.date >= lo) & (p.dt.date < hi)).sum()
        print(f"свечи: {lo} .. {hi}; новостей внутри периода свечей: {inside}")
        if inside == 0:
            print("[warn] периоды НЕ пересекаются: образцов не будет. Загрузите свечи командой `candles`.")

# ============================================ СИНХРОНИЗАЦИЯ ============================================
def find_tickers(text):
    return [t for t, ps in _PATTERNS.items() if any(p.search(text) for p in ps)]


def _norm(t):
    return re.sub(r"[^\w\s]", " ", t.lower()).split()


def dedup(news):
    """Точные дубли (хеш) и почти-дубли (Jaccard по 3-шинглам в окне NEAR_DUP_HOURS); остаётся самая ранняя."""
    news = news.sort_values("published").reset_index(drop=True)
    toks = [_norm(f"{a} {b}") for a, b in zip(news.title, news.text)]
    sh = [set(zip(t, t[1:], t[2:])) or set(t) for t in toks]
    h = [hashlib.md5(" ".join(t).encode()).hexdigest() for t in toks]
    pub = news["published"].tolist()
    reason, seen, kept = [], set(), []
    for i in range(len(news)):
        if h[i] in seen:
            reason.append("exact_dup"); continue
        dup = False
        for j in reversed(kept):
            if (pub[i] - pub[j]).total_seconds() > NEAR_DUP_HOURS * 3600:
                break
            u = len(sh[i] | sh[j])
            if u and len(sh[i] & sh[j]) / u >= NEAR_DUP_JACCARD:
                dup = True; break
        if dup:
            reason.append("near_dup")
        else:
            reason.append(""); seen.add(h[i]); kept.append(i)
    news["dup_reason"] = reason
    return news


def trading_calendar(candles):
    """Торговый день = дата, в которую есть свеча хотя бы у одного тикера."""
    return sorted(set(candles["date"]))


def close_dt(d):
    return datetime.combine(d, dtime(*CLOSE_HHMM), tzinfo=MSK)


def open_dt(d):
    return datetime.combine(d, dtime(*OPEN_HHMM), tzinfo=MSK)


def anchor_date(t, cal):
    """Последний торговый день, чья сессия закрыта к моменту t."""
    d = t.date()
    i = bisect.bisect_right(cal, d)
    if i and cal[i - 1] == d and t >= close_dt(d):
        return d
    j = bisect.bisect_left(cal, d)
    return cal[j - 1] if j else None


def session_kind(t, calset):
    d = t.date()
    if d not in calset:
        return "non_trading_day"
    if t < open_dt(d):
        return "pre_open"
    return "after_close" if t >= close_dt(d) else "intraday"


def build_samples(news, candles, drop_intraday=False):
    """news: уже очищенные новости (колонки tickers, ...). Возвращает (samples, windows, exclusions)."""
    cal = trading_calendar(candles)
    calset, pos = set(cal), {d: i for i, d in enumerate(cal)}
    per = {}
    for t, g in candles.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        per[t] = (g, {d: i for i, d in enumerate(g["date"])})
    samples, windows, excl = [], {}, []
    for r in news.itertuples():
        ticks = r.tickers
        kind = session_kind(r.published, calset)
        a = anchor_date(r.published, cal)
        if drop_intraday and kind == "intraday":
            excl.append((r.url, ",".join(ticks), "intraday_news", 1)); continue
        for t in ticks:
            def bad(reason):
                excl.append((r.url, t, reason, 1))
            if t not in per:
                bad("no_market_data"); continue
            if a is None:
                bad("before_calendar"); continue
            g, idx = per[t]
            if a not in idx:
                bad("no_bar_on_anchor"); continue
            i = idx[a]
            if i + 1 < WINDOW:
                bad("short_window"); continue
            w = g.iloc[i - WINDOW + 1: i + 1]
            if pos[w["date"].iloc[-1]] - pos[w["date"].iloc[0]] != WINDOW - 1:
                bad("calendar_gap_in_window"); continue
            if w[["open", "high", "low", "close"]].isna().any().any():
                bad("nan_in_window"); continue
            if pos[a] + HORIZON >= len(cal):
                bad("no_future_calendar"); continue      # новость слишком свежая: следующего торгового дня ещё нет
            a1 = cal[pos[a] + HORIZON]
            if a1 not in idx:
                bad("no_target_bar"); continue
            if (a1 - a).days > MAX_GAP_DAYS:
                bad("long_market_gap"); continue          # цель не «следующий день», а возврат после закрытия рынка
            ret = g["close"].iloc[idx[a1]] / g["close"].iloc[i] - 1
            if abs(ret) < 1e-12:
                bad("flat_return"); continue
            sid = f"{t}_{r.published:%Y%m%dT%H%M%S}_{hashlib.md5(r.url.encode()).hexdigest()[:8]}"
            samples.append({"sample_id": sid, "ticker": t, "source": r.source, "published": r.published,
                            "session": kind, "anchor_date": a, "target_date": a1,
                            "text": f"{r.title}. {r.text}".strip(), "n_tickers": len(ticks),
                            "ret_next": ret, "label": int(ret > 0),
                            "volume_zero_in_window": int((w["volume"] == 0).any()),
                            "image": f"img/{t}_{a}.png"})
            windows[(t, a)] = w.reset_index(drop=True)
    cols = ["sample_id", "ticker", "source", "published", "session", "anchor_date", "target_date", "text",
            "n_tickers", "ret_next", "label", "volume_zero_in_window", "image"]
    return (pd.DataFrame(samples, columns=cols), windows,
            pd.DataFrame(excl, columns=["url", "ticker", "reason", "n"]))


def leak_check(samples, windows):
    """0 = нарушений нет: окно кончается на якоре, якорь закрыт к моменту новости, цель строго позже якоря."""
    bad = 0
    for s in samples.itertuples():
        w = windows[(s.ticker, s.anchor_date)]
        ok = (len(w) == WINDOW and w["date"].max() == s.anchor_date and w["date"].is_monotonic_increasing
              and close_dt(s.anchor_date) <= s.published and s.target_date > s.anchor_date)
        bad += (not ok)
    return bad


def temporal_split(samples, cal, split=SPLIT, embargo=EMBARGO_DAYS):
    """Хронологическое разбиение по якорной дате с зазором (embargo) в торговых днях между частями."""
    s = samples.sort_values(["anchor_date", "published"]).reset_index(drop=True)
    n = len(s)
    c1, c2 = int(n * split[0]), min(int(n * (split[0] + split[1])), n - 1)
    d_val, d_test = s.anchor_date[c1], s.anchor_date[c2]
    pos = {d: i for i, d in enumerate(cal)}
    shift = lambda d, k: cal[max(0, min(len(cal) - 1, pos[d] + k))]
    tr_end, va_end = shift(d_val, -embargo), shift(d_test, -embargo)
    part = []
    for d in s.anchor_date:
        part.append("train" if d < tr_end else "embargo" if d < d_val else "val" if d < va_end
                    else "embargo" if d < d_test else "test")
    s["split"] = part
    return s


# ============================================ ИЗОБРАЖЕНИЯ ============================================
def render(window, path, size=224):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, av) = plt.subplots(2, 1, figsize=(size / 100, size / 100), dpi=100,
                                 gridspec_kw={"height_ratios": [4, 1], "hspace": 0.02})
    for i, r in enumerate(window.itertuples()):
        c = "#2e7d32" if r.close >= r.open else "#c62828"
        ax.vlines(i, r.low, r.high, color=c, lw=0.8)
        ax.add_patch(plt.Rectangle((i - 0.3, min(r.open, r.close)), 0.6, abs(r.close - r.open) or 1e-9, color=c))
    av.bar(np.arange(len(window)), window["volume"].fillna(0), color="#607d8b", width=0.6)
    for a in (ax, av):
        a.set_xlim(-1, len(window)); a.axis("off")
    ax.autoscale_view()
    fig.savefig(path)
    plt.close(fig)


def validate_image(path, min_std=2.0, size=224):
    """'' если файл в порядке, иначе причина."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("L")
            if im.size != (size, size):
                return "bad_size"
            if float(np.asarray(im).std()) < min_std:
                return "blank_image"
    except Exception as e:
        return f"unreadable:{type(e).__name__}"
    return ""


# ============================================ ОТЧЁТ И СБОРКА ============================================
def md(df):
    try:
        return df.to_markdown()
    except ImportError:                               # нет tabulate
        return "```\n" + df.to_string() + "\n```"


def qd(s):
    return s.describe(percentiles=[.05, .5, .95]).round(2).to_dict()


def write_report(out, news_raw, n_kept, cal, samples, excl, split, bad_img, leaks, dupc):
    L = ["# Отчёт о данных", "",
         "## 1. Объёмы", f"- новостей до очистки: {len(news_raw)}",
         f"- без разобранной даты: {int(news_raw['published'].isna().sum())}",
         f"- без указания часового пояса (принят MSK): {int(news_raw['naive_tz'].sum())}",
         f"- упоминают тикер и остались после удаления дубликатов: {n_kept}",
         f"- дублей свечей удалено: {dupc}",
         f"- торговых дней в календаре: {len(cal)} ({cal[0]}..{cal[-1]})",
         f"- итоговых образцов (новость x тикер): {len(samples)}",
         f"- нарушений причинности (leak_check): {leaks}", ""]
    L += ["## 2. Причины исключения", md(excl.groupby("reason")["n"].sum().sort_values(ascending=False).to_frame("n")), ""]
    if len(samples):
        L += ["## 3. По тикерам", md(samples.groupby("ticker").agg(n=("label", "size"), up_share=("label", "mean")).round(3)), "",
              "## 4. По источникам", md(samples["source"].value_counts().to_frame("n")), "",
              "## 5. По месяцам", md(samples.groupby(pd.to_datetime(samples.anchor_date).dt.to_period("M")).size().to_frame("n")), "",
              "## 6. Время публикации (MSK)", md(samples["session"].value_counts(normalize=True).round(3).to_frame("share")),
              f"- по часам: {samples.published.dt.hour.value_counts().sort_index().to_dict()}", "",
              "## 7. Целевая переменная", f"- доля роста: {samples.label.mean():.3f}", f"- ret_next: {qd(samples.ret_next)}", "",
              "## 8. Длина текста (символы)", f"{qd(samples.text.str.len())}", "",
              "## 9. Качество рынка", f"- образцов с нулевым объёмом в окне: {int(samples.volume_zero_in_window.sum())}",
              f"- уникальных изображений: {samples.image.nunique()}, битых/пустых: {bad_img}", ""]
        sp = split.groupby("split").agg(n=("label", "size"), start=("anchor_date", "min"), end=("anchor_date", "max"),
                                        up_share=("label", "mean")).round(3)
        L += ["## 10. Разбиение", md(sp), ""]
    open(os.path.join(out, "data_report.md"), "w", encoding="utf-8").write("\n".join(L))


def build(candles, news_raw, out="out", drop_intraday=False):
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    dupc = int(candles.duplicated(["ticker", "date"]).sum())
    candles = candles.drop_duplicates(["ticker", "date"])
    cal = trading_calendar(candles)

    news = news_raw.dropna(subset=["published"]).copy()
    news["tickers"] = [find_tickers(f"{a} {b}") for a, b in zip(news.title, news.text)]
    excl = [("(агрегировано)", "", "no_date", len(news_raw) - len(news))]
    has = news["tickers"].map(len) > 0
    excl.append(("(агрегировано)", "", "no_ticker", int((~has).sum())))
    news = dedup(news[has])
    for r in news[news.dup_reason != ""].itertuples():
        excl.append((r.url, ",".join(r.tickers), r.dup_reason, 1))
    news = news[news.dup_reason == ""]
    many = news["tickers"].map(len) > MAX_TICKERS_PER_NEWS
    for r in news[many].itertuples():
        excl.append((r.url, ",".join(r.tickers), "too_many_tickers", 1))
    news = news[~many]

    samples, windows, ex2 = build_samples(news, candles, drop_intraday)
    excl = pd.concat([pd.DataFrame(excl, columns=ex2.columns), ex2], ignore_index=True)
    excl = excl[excl["n"] > 0]
    excl.to_csv(os.path.join(out, "exclusions.csv"), index=False)
    if samples.empty:
        write_report(out, news_raw, len(news), cal, samples, excl, None, 0, 0, dupc)
        print("[!] Образцов 0. Причины исключения:\n" + excl.groupby("reason")["n"].sum().sort_values(ascending=False).to_string())
        print("Типичные причины: новости не пересекаются со свечами по датам (no_future_calendar/before_calendar), "
              "нет упоминаний тикеров (no_ticker), слишком короткая история свечей (short_window). См. инструкцию, п. 3.")
        return None
    leaks = leak_check(samples, windows)
    bad_rows = []
    for (t, a), w in windows.items():                      # одна картинка на (тикер, якорь)
        p = os.path.join(out, "img", f"{t}_{a}.png")
        render(w, p)
        r = validate_image(p)
        if r:
            bad_rows.append((f"{t}_{a}", r))
    bad_ids = {b[0] for b in bad_rows}
    pd.DataFrame(bad_rows, columns=["image_key", "reason"]).to_csv(os.path.join(out, "bad_images.csv"), index=False)
    samples = samples[~samples["image"].str.replace("img/", "").str.replace(".png", "").isin(bad_ids)]
    split = temporal_split(samples, cal)
    split.to_csv(os.path.join(out, "samples.csv"), index=False)
    write_report(out, news_raw, len(news), cal, samples, excl, split, len(bad_rows), leaks, dupc)
    print(f"samples={len(samples)} images={samples.image.nunique()} leaks={leaks} bad_images={len(bad_rows)}")
    print(split["split"].value_counts().to_string())
    print(f"Отчёт: {os.path.join(out, 'data_report.md')}")
    return split


def cmd_check(a):
    print(f"python {sys.version.split()[0]}; pandas {pd.__version__}")
    for m in ("matplotlib", "PIL", "tabulate", "pyarrow"):
        try:
            __import__(m); print(f"  {m}: OK")
        except ImportError:
            print(f"  {m}: НЕТ  (pip install matplotlib pillow tabulate pyarrow)")
    try:
        df = fetch_candles("SBER", str(pd.Timestamp.today().date() - pd.Timedelta(days=14)),
                           str(pd.Timestamp.today().date()))
        print(f"MOEX ISS: OK ({len(df)} дневных свечей SBER за 2 недели)")
    except Exception as e:
        print(f"MOEX ISS: FAIL - {e}")
    try:
        r = requests.head(LENTA_URL, headers=HEADERS, timeout=30, allow_redirects=True)
        r.raise_for_status()
        print(f"Hugging Face (lenta-news): OK, размер файла {int(r.headers.get('content-length', 0)) / 1e6:.0f} МБ")
    except Exception as e:
        print(f"Hugging Face: FAIL - {type(e).__name__}: {e}\n  скачайте train.parquet вручную и запустите: news --file путь")


def cmd_candles(a):
    os.makedirs(DATA_DIR, exist_ok=True)
    parts = []
    for t in TICKERS:
        d = fetch_candles(t, a.start, a.end)
        print(f"{t}: {len(d)} свечей" + (f" ({d.date.min()}..{d.date.max()})" if len(d) else " - ПУСТО"))
        parts.append(d)
    if not any(len(p) for p in parts):
        sys.exit("Свечи не загружены: запустите `check` и смотрите причину.")
    pd.concat(parts).to_csv(CANDLES_CSV, index=False)
    print(f"сохранено: {CANDLES_CSV}")


def cmd_news(a):
    path = a.file or LENTA_PARQUET
    if not a.file and not os.path.exists(path):
        print(f"скачиваю {LENTA_URL}")
        try:
            download(LENTA_URL, path)
        except Exception as e:
            sys.exit(f"Не удалось скачать ({type(e).__name__}: {e}). Скачайте train.parquet вручную со страницы набора "
                     f"и запустите: python task2_sync.py news --file путь/к/train.parquet")
    news = read_lenta(path, a.topic)
    p = news["published"]
    news = news[(p.dt.date >= pd.Timestamp(NEWS_START).date()) & (p.dt.date <= pd.Timestamp(NEWS_END).date()) | p.isna()]
    has = [bool(find_tickers(f"{t} {x}")) for t, x in zip(news.title, news.text)]
    print(f"из них упоминают {', '.join(TICKERS)}: {sum(has)}")
    save_news(news)
    print_news_span(news, f"сохранено в {NEWS_CSV}")


def cmd_build(a):
    build(load_candles(), load_news(), a.out, a.drop_intraday)


def cmd_demo(a):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2025-01-01", "2025-12-31").drop(pd.to_datetime(["2025-05-01", "2025-05-09", "2025-06-12"]))
    rows = []
    for t in ["SBER"]:
        p = 100 + rng.normal(0, 1, len(days)).cumsum()
        for d, c in zip(days, p):
            o = c + rng.normal(0, .3)
            rows.append({"ticker": t, "date": d.date(), "open": o, "close": c, "high": max(o, c) + .2,
                         "low": min(o, c) - .2, "volume": float(rng.integers(0, 1000))})
    names, news = {"SBER": "Сбербанк"}, []
    for i in range(400):
        d = days[rng.integers(25, len(days) - 3)] + pd.Timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
        if i % 17 == 0:
            d += pd.Timedelta(days=(5 - d.weekday()) % 7)
        t = list(names)[i % 1]
        news.append({"source": "demo", "title": f"{names[t]} сообщил о результатах за квартал номер {i}",
                     "text": "Подробности отчётности компании.", "url": f"http://demo/{i}",
                     "published": pd.Timestamp(d).tz_localize(MSK), "naive_tz": i % 11 == 0})
        if i % 13 == 0:
            news.append({**news[-1], "url": f"http://demo/{i}b"})
    build(pd.DataFrame(rows), pd.DataFrame(news), "out_demo")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", help="другой тикер вместо SBER (до подкоманды), напр. GAZP; см. PRESETS")
    ap.add_argument("--aliases", help="regex-алиасы через ';' (если нет заготовки в PRESETS)")
    ap.add_argument("--data-dir", help="папка для свечей этого тикера, напр. data_gazp (data/ от SBER не трогается)")
    ap.add_argument("--news-csv", help="готовый news.csv, напр. data/news.csv от SBER: перекачивать новости не нужно")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check", help="проверить пакеты и доступ к MOEX и Hugging Face").set_defaults(f=cmd_check)
    p = sp.add_parser("candles", help="скачать дневные свечи тикера (по умолчанию SBER)"); p.set_defaults(f=cmd_candles)
    p.add_argument("--start", default=CANDLES_START); p.add_argument("--end", default=CANDLES_END)
    p = sp.add_parser("news", help="Lenta-news: скачать и отобрать раздел"); p.set_defaults(f=cmd_news)
    p.add_argument("--file", help="локальный train.parquet (или csv), если автоскачивание недоступно")
    p.add_argument("--topic", default=TOPIC)
    p = sp.add_parser("build", help="собрать образцы, изображения, отчёт"); p.set_defaults(f=cmd_build)
    p.add_argument("--out", default="out"); p.add_argument("--drop-intraday", action="store_true")
    sp.add_parser("demo", help="проверка на синтетике").set_defaults(f=cmd_demo)
    a = ap.parse_args()
    configure(a.ticker, a.aliases, a.data_dir, a.news_csv)
    a.f(a)


if __name__ == "__main__":
    main()
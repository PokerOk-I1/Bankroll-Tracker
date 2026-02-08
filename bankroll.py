#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bankroll.py — учёт и аналитика банкролла
"""

import argparse
import os
import sys
import uuid
import textwrap
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_DB_PATH = "bankroll_data.csv"
DEFAULT_ROOM = "PokerOK"
DEFAULT_CURRENCY = "USD"

COLUMNS = [
    "id",
    "date",
    "room",
    "game_type",
    "limit",
    "format",
    "buyin",
    "fee",
    "rebuys",
    "addons",
    "total_buyin",
    "cashout",
    "result",
    "currency",
    "tags",
    "notes",
]


# ---------- Утилиты ----------

def verbose_print(args, msg):
    if getattr(args, "verbose", False):
        print(f"[DEBUG] {msg}")


def parse_date_string(date_str):
    """Парсит строку в datetime. Ожидает 'YYYY-MM-DD' или 'YYYY-MM-DD HH:MM'."""
    if date_str is None or pd.isna(date_str):
        return None
    if isinstance(date_str, datetime):
        return date_str
    s = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Попробуем ISO-парсер
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def load_data(db_path, args=None, require_exists=True):
    """Загружает CSV с данными. Если файла нет и require_exists=True — выбрасывает FileNotFoundError."""
    if not os.path.exists(db_path):
        if require_exists:
            raise FileNotFoundError(db_path)
        else:
            return pd.DataFrame(columns=COLUMNS)

    if args:
        verbose_print(args, f"Чтение базы данных из {db_path}")

    df = pd.read_csv(db_path, encoding="utf-8")

    # Убедимся, что есть все нужные колонки
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Приводим к нужному порядку колонок
    df = df[COLUMNS]

    # Преобразуем числовые поля
    numeric_cols = ["buyin", "fee", "rebuys", "addons", "total_buyin", "cashout", "result"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Преобразуем дату
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def save_data(df, db_path, args=None):
    """Сохраняет DataFrame в CSV."""
    if args:
        verbose_print(args, f"Сохранение базы данных в {db_path}")
    df.to_csv(db_path, index=False, encoding="utf-8")


def calc_total_buyin(row):
    """Вычисляет общий бай-ин (с ребаями и аддонами)."""
    buyin = row.get("buyin") or 0.0
    fee = row.get("fee") or 0.0
    rebuys = row.get("rebuys") or 0
    addons = row.get("addons") or 0
    try:
        rebuys = int(rebuys)
    except Exception:
        rebuys = 0
    try:
        addons = int(addons)
    except Exception:
        addons = 0

    return (buyin + fee) * (1 + rebuys + addons)


def calc_result(row):
    """Вычисляет результат = cashout - total_buyin."""
    cashout = row.get("cashout") or 0.0
    total_buyin = row.get("total_buyin") or 0.0
    return cashout - total_buyin


def calc_max_drawdown(cum_profit):
    """
    Максимальный даунсвинг по серии кумулятивного профита.
    cum_profit — pandas.Series.
    """
    if cum_profit.empty:
        return 0.0
    peak = cum_profit.iloc[0]
    max_dd = 0.0
    for x in cum_profit:
        if x > peak:
            peak = x
        drawdown = x - peak
        if drawdown < max_dd:
            max_dd = drawdown
    return max_dd


def apply_filters(df, args):
    """Применяет фильтры к DataFrame по аргументам CLI."""

    if df.empty:
        return df

    filt = pd.Series([True] * len(df), index=df.index)

    if args.from_date:
        from_dt = parse_date_string(args.from_date)
        if from_dt:
            filt &= df["date"] >= from_dt

    if args.to_date:
        to_dt = parse_date_string(args.to_date)
        if to_dt:
            # включительно
            filt &= df["date"] <= to_dt

    if args.room:
        filt &= df["room"].astype(str).str.lower() == args.room.lower()

    if args.game_type:
        # может быть несколько через запятую
        types = [t.strip().lower() for t in args.game_type.split(",")]
        filt &= df["game_type"].astype(str).str.lower().isin(types)

    if args.format:
        formats = [f.strip().lower() for f in args.format.split(",")]
        filt &= df["format"].astype(str).str.lower().isin(formats)

    if args.limit:
        substring = args.limit.lower()
        filt &= df["limit"].astype(str).str.lower().str.contains(substring)

    if args.tag:
        tag = args.tag.lower()
        filt &= df["tags"].fillna("").astype(str).str.lower().str.contains(tag)

    return df[filt].copy()


def format_money(x, currency=DEFAULT_CURRENCY):
    if pd.isna(x):
        x = 0.0
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f} {currency}"


def print_error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


# ---------- Команды ----------

def cmd_add(args):
    db_path = args.db_path or DEFAULT_DB_PATH

    df = load_data(db_path, args, require_exists=False)

    # Подготовим новую запись
    rec = {}

    rec["id"] = str(uuid.uuid4())

    if args.date:
        date = parse_date_string(args.date)
        if not date:
            print_error("Неверный формат даты. Ожидается YYYY-MM-DD или YYYY-MM-DD HH:MM.")
            sys.exit(1)
    else:
        date = datetime.now()

    rec["date"] = date

    rec["room"] = args.room or DEFAULT_ROOM
    rec["game_type"] = args.game_type or ""
    rec["limit"] = args.limit or ""
    rec["format"] = args.format or ""
    rec["currency"] = args.currency or DEFAULT_CURRENCY
    rec["tags"] = args.tags or ""
    rec["notes"] = args.notes or ""

    def to_float(v, default=0.0):
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    def to_int(v, default=0):
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    rec["buyin"] = to_float(args.buyin, 0.0)
    rec["fee"] = to_float(args.fee, 0.0)
    rec["rebuys"] = to_int(args.rebuys, 0)
    rec["addons"] = to_int(args.addons, 0)
    rec["cashout"] = to_float(args.cashout, 0.0)

    if args.total_buyin is not None:
        rec["total_buyin"] = to_float(args.total_buyin, 0.0)
    else:
        rec["total_buyin"] = calc_total_buyin(rec)

    if args.result is not None:
        rec["result"] = to_float(args.result, 0.0)
    else:
        rec["result"] = calc_result(rec)

    # Добавляем запись
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    save_data(df, db_path, args)

    print(
        f"[OK] Добавлена запись: profit = {format_money(rec['result'], rec['currency'])} "
        f"(cashout={rec['cashout']:.2f}, total_buyin={rec['total_buyin']:.2f})"
    )


def cmd_import(args):
    db_path = args.db_path or DEFAULT_DB_PATH
    import_path = args.file

    if not os.path.exists(import_path):
        print_error(f"Файл {import_path} не найден.")
        sys.exit(1)

    try:
        src_df = pd.read_csv(import_path, encoding="utf-8")
    except Exception as e:
        print_error(f"Не удалось прочитать файл {import_path}: {e}")
        sys.exit(1)

    df = load_data(db_path, args, require_exists=False)

    # Маппинг колонок импортируемого файла к нашим
    # Попытаемся угадать по названиям
    col_map = {
        "date": ["date", "Date", "datetime", "time"],
        "buyin": ["buyin", "buy_in", "buy-in", "bi"],
        "fee": ["fee", "rake", "commission"],
        "cashout": ["cashout", "payout", "prize", "winnings", "win"],
        "room": ["room", "site", "room_name"],
        "game_type": ["game_type", "type", "gameType"],
        "limit": ["limit", "stakes", "stake"],
        "format": ["format", "game", "variant"],
        "rebuys": ["rebuys", "rebuy_count"],
        "addons": ["addons", "addon_count"],
        "currency": ["currency", "curr"],
        "tags": ["tags", "label"],
        "notes": ["notes", "comment", "description"],
    }

    def find_col(target):
        candidates = col_map.get(target, [])
        for cand in candidates:
            for real in src_df.columns:
                if real.lower() == cand.lower():
                    return real
        return None

    # Проверим обязательные поля
    date_col = find_col("date")
    buyin_col = find_col("buyin")
    cashout_col = find_col("cashout")
    # fee не обязателен, может быть 0

    if not date_col or not buyin_col or not cashout_col:
        print_error(
            "Не удалось сопоставить обязательные колонки (date, buyin, cashout). "
            "Проверьте названия колонок в импортируемом CSV."
        )
        sys.exit(1)

    imported_records = []
    errors = 0

    for idx, row in src_df.iterrows():
        try:
            rec = {}
            rec["id"] = str(uuid.uuid4())

            d = parse_date_string(row[date_col])
            if not d:
                raise ValueError("Неверная дата")

            rec["date"] = d

            # Опциональные поля
            rec["room"] = row[find_col("room")] if find_col("room") else DEFAULT_ROOM
            rec["game_type"] = row[find_col("game_type")] if find_col("game_type") else ""
            rec["limit"] = row[find_col("limit")] if find_col("limit") else ""
            rec["format"] = row[find_col("format")] if find_col("format") else ""
            rec["currency"] = row[find_col("currency")] if find_col("currency") else DEFAULT_CURRENCY
            rec["tags"] = row[find_col("tags")] if find_col("tags") else ""
            rec["notes"] = row[find_col("notes")] if find_col("notes") else ""

            def safe_float(x, default=0.0):
                try:
                    return float(x)
                except Exception:
                    return default

            def safe_int(x, default=0):
                try:
                    return int(x)
                except Exception:
                    return default

            rec["buyin"] = safe_float(row[buyin_col], 0.0)

            fee_col = find_col("fee")
            rec["fee"] = safe_float(row[fee_col], 0.0) if fee_col else 0.0

            rec["cashout"] = safe_float(row[cashout_col], 0.0)

            reb_col = find_col("rebuys")
            add_col = find_col("addons")
            rec["rebuys"] = safe_int(row[reb_col], 0) if reb_col else 0
            rec["addons"] = safe_int(row[add_col], 0) if add_col else 0

            rec["total_buyin"] = calc_total_buyin(rec)
            rec["result"] = calc_result(rec)

            imported_records.append(rec)
        except Exception:
            errors += 1
            continue

    if not imported_records:
        print_error("Не удалось импортировать ни одной записи.")
        sys.exit(1)

    imp_df = pd.DataFrame(imported_records)
    df = pd.concat([df, imp_df], ignore_index=True)
    save_data(df, db_path, args)

    total_profit = imp_df["result"].sum()
    print(f"Импортировано: {len(imported_records)} записей")
    print(f"Пропущено (ошибки): {errors}")
    print(f"Итоговый профит по импортированным: {format_money(total_profit, DEFAULT_CURRENCY)}")


def cmd_summary(args):
    db_path = args.db_path or DEFAULT_DB_PATH

    try:
        df = load_data(db_path, args, require_exists=True)
    except FileNotFoundError:
        print_error(f"Файл базы данных {db_path} не найден. Добавьте записи или импортируйте данные.")
        sys.exit(1)

    df = apply_filters(df, args)

    if df.empty:
        print("По заданным фильтрам записи не найдены.")
        return

    currency = df["currency"].mode().iloc[0] if not df["currency"].isna().all() else DEFAULT_CURRENCY

    total_games = len(df)
    # Сумма вложений и кэшаута
    total_buyin = df["total_buyin"].fillna(0).sum()
    total_cashout = df["cashout"].fillna(0).sum()
    total_profit = df["result"].fillna(0).sum()

    # Дни
    df["date_only"] = df["date"].dt.date
    daily_profit = df.groupby("date_only")["result"].sum().sort_index()

    best_day_profit = daily_profit.max()
    best_day = daily_profit.idxmax()

    worst_day_profit = daily_profit.min()
    worst_day = daily_profit.idxmin()

    cum_profit = daily_profit.cumsum()
    max_dd = calc_max_drawdown(cum_profit)

    # ITM
    itm_count = (df["cashout"] > 0).sum()
    itm_pct = itm_count / total_games * 100 if total_games > 0 else 0.0

    roi = (total_cashout - total_buyin) / total_buyin * 100 if total_buyin > 0 else 0.0

    avg_profit_per_game = total_profit / total_games if total_games > 0 else 0.0

    period_from = df["date"].min().date()
    period_to = df["date"].max().date()

    # Формируем строку с фильтрами
    filters = []
    if args.room:
        filters.append(f"room={args.room}")
    if args.game_type:
        filters.append(f"game_type={args.game_type}")
    if args.format:
        filters.append(f"format={args.format}")
    if args.limit:
        filters.append(f"limit contains '{args.limit}'")
    if args.tag:
        filters.append(f"tag contains '{args.tag}'")
    filters_str = ", ".join(filters) if filters else "нет (использованы все записи)"

    print()
    print(f"Период: {period_from} — {period_to}")
    print(f"Фильтры: {filters_str}")
    print()
    print(f"Общее число игр: {total_games}")
    print(f"Дней с игрой: {len(daily_profit)}")
    print()
    print(f"Общий вложенный бай-ин (total_buyin): {total_buyin:.2f} {currency}")
    print(f"Общая выплата (cashout):           {total_cashout:.2f} {currency}")
    print(f"Итоговый профит:                   {format_money(total_profit, currency)}")
    print()
    print(f"ROI:              {roi:.1f} %")
    print(f"ITM (выплата > 0): {itm_pct:.1f} %")
    print(f"Средний профит на игру: {format_money(avg_profit_per_game, currency)}")
    print()
    print(f"Лучший день:  {best_day} ({format_money(best_day_profit, currency)})")
    print(f"Худший день:  {worst_day} ({format_money(worst_day_profit, currency)})")
    print(f"Макс. даунсвинг за период: {format_money(max_dd, currency)}")
    print()


def cmd_graph(args):
    db_path = args.db_path or DEFAULT_DB_PATH

    try:
        df = load_data(db_path, args, require_exists=True)
    except FileNotFoundError:
        print_error(f"Файл базы данных {db_path} не найден. Добавьте записи или импортируйте данные.")
        sys.exit(1)

    df = apply_filters(df, args)

    if df.empty:
        print("По заданным фильтрам записи не найдены, график не построен.")
        return

    currency = df["currency"].mode().iloc[0] if not df["currency"].isna().all() else DEFAULT_CURRENCY

    df["date_only"] = df["date"].dt.date
    daily_profit = df.groupby("date_only")["result"].sum().sort_index()
    cum_profit = daily_profit.cumsum()

    # Создание графика
    plt.figure()
    plt.plot(daily_profit.index, cum_profit.values, marker="o")
    plt.xlabel("Дата")
    plt.ylabel(f"Кумулятивный профит ({currency})")

    period_from = daily_profit.index.min()
    period_to = daily_profit.index.max()
    title_filters = []
    if args.room:
        title_filters.append(f"room={args.room}")
    if args.game_type:
        title_filters.append(f"game_type={args.game_type}")
    if args.format:
        title_filters.append(f"format={args.format}")
    filters_str = ", ".join(title_filters)
    if filters_str:
        title = f"Bankroll ({period_from} — {period_to}) [{filters_str}]"
    else:
        title = f"Bankroll ({period_from} — {period_to})"
    plt.title(title)
    plt.grid(True)

    output_path = args.output or "bankroll_graph.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[OK] График сохранён в файл: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


def _print_table(df, max_rows=None):
    """Примитивный вывод таблицы в консоль."""
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    columns = ["date", "game_type", "limit", "buyin", "fee", "rebuys", "addons", "cashout", "result", "currency", "notes"]

    headers = {
        "date": "Дата",
        "game_type": "Тип",
        "limit": "Лимит",
        "buyin": "Бай-ин",
        "fee": "Рейк",
        "rebuys": "Ребаи",
        "addons": "Аддоны",
        "cashout": "Cashout",
        "result": "Результат",
        "currency": "Валюта",
        "notes": "Заметка",
    }

    # Подготовим строки как текст
    rows = []
    for _, r in df.iterrows():
        row = []
        row.append(r["date"].strftime("%Y-%m-%d") if not pd.isna(r["date"]) else "")
        row.append(str(r["game_type"] or ""))
        row.append(str(r["limit"] or ""))
        row.append(f"{r['buyin']:.2f}" if pd.notna(r["buyin"]) else "")
        row.append(f"{r['fee']:.2f}" if pd.notna(r["fee"]) else "")
        row.append(str(int(r["rebuys"])) if pd.notna(r["rebuys"]) else "0")
        row.append(str(int(r["addons"])) if pd.notna(r["addons"]) else "0")
        row.append(f"{r['cashout']:.2f}" if pd.notna(r["cashout"]) else "")
        row.append(f"{r['result']:.2f}" if pd.notna(r["result"]) else "")
        row.append(str(r["currency"] or ""))
        # Обрежем заметку, если слишком длинная
        note = str(r["notes"] or "")
        if len(note) > 40:
            note = note[:37] + "..."
        row.append(note)
        rows.append(row)

    # Определяем ширину колонок
    col_widths = []
    for i, col in enumerate(columns):
        header = headers[col]
        max_len = len(header)
        for row in rows:
            max_len = max(max_len, len(str(row[i])))
        col_widths.append(max_len)

    # Печатаем заголовок
    header_line = "  ".join(headers[col].ljust(col_widths[i]) for i, col in enumerate(columns))
    print(header_line)
    print("-" * len(header_line))

    # Печатаем строки
    for row in rows:
        line = "  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(columns)))
        print(line)


def cmd_history(args):
    db_path = args.db_path or DEFAULT_DB_PATH

    try:
        df = load_data(db_path, args, require_exists=True)
    except FileNotFoundError:
        print_error(f"Файл базы данных {db_path} не найден. Добавьте записи или импортируйте данные.")
        sys.exit(1)

    df = apply_filters(df, args)

    if df.empty:
        print("По заданным фильтрам записи не найдены.")
        return

    df = df.sort_values("date")

    _print_table(df, max_rows=args.limit_rows)

    total_profit = df["result"].fillna(0).sum()
    currency = df["currency"].mode().iloc[0] if not df["currency"].isna().all() else DEFAULT_CURRENCY
    print()
    print(f"Итого по выборке: профит = {format_money(total_profit, currency)} (игр: {len(df)})")


def cmd_reset(args):
    db_path = args.db_path or DEFAULT_DB_PATH

    if not os.path.exists(db_path):
        print(f"Файл {db_path} не существует, ничего удалять не нужно.")
        return

    print(f"ВНИМАНИЕ: будет удалён файл {db_path}. Продолжить? (yes/no): ", end="")
    choice = input().strip().lower()
    if choice == "yes":
        try:
            os.remove(db_path)
            print("[OK] База данных удалена.")
        except Exception as e:
            print_error(f"Не удалось удалить файл: {e}")
    else:
        print("Операция отменена.")


def cmd_config(args):
    db_path = args.db_path or DEFAULT_DB_PATH
    print("Текущая конфигурация:")
    print(f"  Путь к базе данных: {os.path.abspath(db_path)}")
    print(f"  Дефолтный рум: {DEFAULT_ROOM}")
    print(f"  Дефолтная валюта: {DEFAULT_CURRENCY}")


# ---------- Парсинг аргументов и main ----------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Учет и анализ банкролла (PokerOK/GGPoker и др.).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Примеры:
              Добавить запись:
                python bankroll.py add --date 2025-12-05 --game-type MTT --limit "2$" \\
                    --format Holdem --buyin 2 --fee 0.2 --rebuys 1 --cashout 15 \\
                    --tags "evening,reg" --notes "Bounty Hunters $2.10"

              Импортировать CSV:
                python bankroll.py import --file pokerok_results.csv

              Сводка по MTT:
                python bankroll.py summary --game-type MTT --from-date 2025-12-01

              Построить график:
                python bankroll.py graph --from-date 2025-11-01 --to-date 2025-12-06 \\
                    --output bankroll_nov_dec.png

              История:
                python bankroll.py history --from-date 2025-12-01 --limit-rows 20
            """
        ),
    )

    parser.add_argument(
        "--db-path",
        help=f"Путь к CSV файлу с базой (по умолчанию: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод (отладочная информация).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- add ---
    p_add = subparsers.add_parser("add", help="Добавить одну запись (турнир/сессию) вручную.")
    p_add.add_argument("--date", help="Дата/время в формате YYYY-MM-DD или YYYY-MM-DD HH:MM (по умолчанию — сейчас).")
    p_add.add_argument("--room", help="Покер-рум (по умолчанию PokerOK).")
    p_add.add_argument("--game-type", help="Тип игры (например: MTT, Cash, Spin&Go).")
    p_add.add_argument("--limit", help="Лимит/бай-ин (например: 'NL10', '2$').")
    p_add.add_argument("--format", help="Формат (Holdem, Omaha, Short Deck и т.п.).")
    p_add.add_argument("--buyin", type=float, help="Бай-ин (без рейка).")
    p_add.add_argument("--fee", type=float, help="Рейк.")
    p_add.add_argument("--rebuys", type=int, help="Количество ребаев.")
    p_add.add_argument("--addons", type=int, help="Количество аддонов.")
    p_add.add_argument("--total-buyin", type=float, help="Общий вложенный бай-ин (если не указано — считается).")
    p_add.add_argument("--cashout", type=float, help="Выплата (0, если ничего не выиграл).")
    p_add.add_argument("--result", type=float, help="Итоговый результат (если не указано — считается).")
    p_add.add_argument("--currency", help="Валюта (по умолчанию USD).")
    p_add.add_argument("--tags", help="Теги через запятую (например: 'evening,reg').")
    p_add.add_argument("--notes", help="Заметка.")

    p_add.set_defaults(func=cmd_add)

    # --- import ---
    p_import = subparsers.add_parser("import", help="Импортировать несколько записей из CSV.")
    p_import.add_argument("--file", required=True, help="Путь к CSV файлу с результатами.")
    p_import.set_defaults(func=cmd_import)

    # Общие фильтры (для summary, graph, history)
    def add_common_filters(p):
        p.add_argument("--from-date", help="Фильтр: начиная с даты (YYYY-MM-DD).")
        p.add_argument("--to-date", help="Фильтр: до даты (YYYY-MM-DD).")
        p.add_argument("--room", help="Фильтр по руму.")
        p.add_argument("--game-type", help="Фильтр по типу игры (можно несколько через запятую).")
        p.add_argument("--format", help="Фильтр по формату (Holdem,Omaha,...).")
        p.add_argument("--limit", help="Фильтр по лимиту/ставке (подстрока).")
        p.add_argument("--tag", help="Фильтр по тегу (подстрока в поле tags).")

    # --- summary ---
    p_summary = subparsers.add_parser("summary", help="Вывести сводку статистики.")
    add_common_filters(p_summary)
    p_summary.set_defaults(func=cmd_summary)

    # --- graph ---
    p_graph = subparsers.add_parser("graph", help="Построить график кумулятивного профита.")
    add_common_filters(p_graph)
    p_graph.add_argument("--output", help="Имя файла для сохранения графика (по умолчанию bankroll_graph.png).")
    p_graph.add_argument("--show", action="store_true", help="Показать график на экране.")
    p_graph.set_defaults(func=cmd_graph)

    # --- history ---
    p_hist = subparsers.add_parser("history", help="Показать историю игр (таблица).")
    add_common_filters(p_hist)
    p_hist.add_argument("--limit-rows", type=int, help="Ограничить количество показанных строк.")
    p_hist.set_defaults(func=cmd_history)

    # --- reset ---
    p_reset = subparsers.add_parser("reset", help="Очистить базу данных (удалить CSV).")
    p_reset.set_defaults(func=cmd_reset)

    # --- config ---
    p_config = subparsers.add_parser("config", help="Показать текущие настройки.")
    p_config.set_defaults(func=cmd_config)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем.")


if __name__ == "__main__":
    main()
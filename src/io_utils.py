"""Задания 1–3: чтение и запись табличных данных."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from src.config import timestamped_filename


def read_table(path: str) -> pd.DataFrame:
    """Универсальное чтение табличных данных (.xlsx, .csv, .txt, .mat)."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".xlsx":
        df = pd.read_excel(p)
    elif ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".txt":
        df = pd.read_csv(p, sep=r"\s+", engine="python")
    elif ext == ".mat":
        mat = loadmat(p)
        arrays = {k: v for k, v in mat.items() if not k.startswith("__")}
        key = next(iter(arrays))
        df = pd.DataFrame(arrays[key])
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    if all(isinstance(c, int) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


def write_table(df: pd.DataFrame, base_name: str) -> dict:
    """Сохраняет DataFrame во все форматы с timestamp в имени."""
    paths = {}

    paths["xlsx"] = timestamped_filename(base_name, "xlsx")
    df.to_excel(paths["xlsx"], index=False, engine="openpyxl")

    paths["csv"] = timestamped_filename(base_name, "csv")
    df.to_csv(paths["csv"], index=False)

    paths["txt"] = timestamped_filename(base_name, "txt")
    df.to_csv(paths["txt"], sep="\t", index=False)

    paths["mat"] = timestamped_filename(base_name, "mat")
    savemat(paths["mat"], {
        "data": df.to_numpy(),
        "columns": np.array(df.columns, dtype=object),
    })

    for fmt, p in paths.items():
        print(f"[io] Сохранено ({fmt}): {p}")

    return paths


def check_data_quality(df: pd.DataFrame) -> dict:
    """Задание 4: проверка пустых ячеек, некорректных данных, типов.

    Использует df.iloc[:, i] вместо df[col], потому что при дублирующихся
    именах столбцов df[col] возвращает DataFrame, а не Series.
    """
    report = {
        "missing": {},
        "dtypes": {},
        "non_numeric": {},
    }

    for i, col in enumerate(df.columns):
        s = df.iloc[:, i]
        key = f"{col} [{i}]"
        report["missing"][key] = int(s.isna().sum())
        report["dtypes"][key] = str(s.dtype)

        converted = pd.to_numeric(s, errors="coerce")
        bad = s.notna() & converted.isna()
        if bad.any():
            report["non_numeric"][key] = s[bad].astype(str).unique().tolist()[:10]

    return report


def print_quality_report(df: pd.DataFrame) -> None:
    """Красивый вывод отчёта о качестве данных."""
    report = check_data_quality(df)
    print("\n=== ОТЧЁТ О КАЧЕСТВЕ ДАННЫХ ===")

    print("\nПропуски (NaN) — только столбцы с пропусками:")
    any_missing = False
    for col, n in report["missing"].items():
        if n > 0:
            print(f"  {col}: {n}")
            any_missing = True
    if not any_missing:
        print("  нет")

    print("\nТипы данных (первые 12):")
    for col, dt in list(report["dtypes"].items())[:12]:
        print(f"  {col}: {dt}")

    print("\nНекорректные (не числовые) значения:")
    if report["non_numeric"]:
        for col, vals in report["non_numeric"].items():
            print(f"  {col}: {vals}")
    else:
        print("  не обнаружено")
    print("=" * 40)


def show_table(df: pd.DataFrame, n: int = 10) -> None:
    """Задание 5: вывод таблицы на экран."""
    print(f"\n=== Первые {n} строк ===")
    print(df.head(n))
    print(f"\nShape: {df.shape}")
    print(f"Dtypes (первые 12):")
    for col, dt in list(df.dtypes.items())[:12]:
        print(f"  {col}: {dt}")


def extract_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Задание 6: извлечение нужных столбцов (копия, исходник не меняется)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    return df[cols].copy()


def cast_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """Задание 7: явные типы данных Pandas.

    Использует безопасное приведение через pd.to_numeric(errors='coerce'),
    чтобы строки вида 'ERROR' или '=ROW()-11' превращались в NaN, а не падали.
    """
    df = df.copy()
    for col, dtype in type_map.items():
        if dtype in ("float64", "float32", "Float64", "int64", "Int64"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if dtype in ("int64", "Int64"):
                df[col] = df[col].astype(dtype)
        else:
            df[col] = df[col].astype(dtype)
    return df
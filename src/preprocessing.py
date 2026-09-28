"""Задания 8–13: предобработка данных, масштабирование, разбиение."""
import numpy as np
import pandas as pd


# ============ Задание 8. Обработка пропусков ============
def handle_missing(df: pd.DataFrame, col: str, action: str, value=None) -> pd.DataFrame:
    """Обработка пропусков и некорректных данных в столбце col.

    action: drop | cast | mean | ffill | bfill | zero | value
    """
    df = df.copy()
    if action == "drop":
        df = df.dropna(subset=[col])
    elif action == "cast":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif action == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif action == "ffill":
        df[col] = df[col].ffill()
    elif action == "bfill":
        df[col] = df[col].bfill()
    elif action == "zero":
        df[col] = df[col].fillna(0)
    elif action == "value":
        df[col] = df[col].fillna(value)
    else:
        raise ValueError(f"Неизвестное действие: {action}")
    return df


def auto_clean(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Автоматическая очистка: cast -> mean."""
    for col in cols:
        df = handle_missing(df, col, "cast")
        df = handle_missing(df, col, "mean")
    return df


# ============ Задание 9. DataFrame -> NumPy ============
def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=np.float64)


# ============ Задания 10–11. Масштабирование ============
class MinMaxScalerCustom:
    """MinMax-нормализация в произвольных границах [a, b]."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a, self.b = a, b
        self.min_, self.max_ = None, None

    def fit(self, x: np.ndarray):
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return self.a + (x - self.min_) * (self.b - self.a) / denom

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return (x_new - self.a) * denom / (self.b - self.a) + self.min_


class StandardScalerCustom:
    """Стандартизация (z-score)."""

    def __init__(self):
        self.mean_, self.std_ = None, None

    def fit(self, x: np.ndarray):
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1, self.std_)
        return (x - self.mean_) / std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1, self.std_)
        return x_new * std + self.mean_


# ============ Задание 12. Разбиение на 3 части ============
def split_train_val_test(x: np.ndarray, ratios, percent: bool = False):
    """Разбивает массив на train/val/test в заданном соотношении."""
    r = np.array(ratios, dtype=float)
    if percent:
        r = r / 100.0
    r = r / r.sum()
    n = len(x)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))
    n_test = n - n_train - n_val

    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    return x_train, x_val, x_test


# ============ Задания 36–37. Скользящее окно и среднее ============
def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    """Возвращает матрицу окон формы (n - width + 1, width)."""
    if width <= 0 or width > len(x):
        raise ValueError("Некорректная ширина окна")
    n = len(x)
    idx = np.arange(width)[None, :] + np.arange(n - width + 1)[:, None]
    return x[idx]


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    w = sliding_window(x, width)
    return w.mean(axis=1)
"""Задания 14–37: графики, статистика, интерполяция, маски."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import spectrogram, periodogram
from scipy.interpolate import interp1d, UnivariateSpline

from src.config import save_figure


# ============ Задание 14. График исходных данных ============
def plot_series(data: np.ndarray, title: str = "Исходные данные"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(data.shape[1]):
        ax.plot(data[:, i], label=f"col_{i}")
    ax.set_title(title)
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "task14_series")
    plt.close(fig)
    return fig


# ============ Задание 15. Гистограмма ============
def plot_histogram(x: np.ndarray, bins: int = 30, name: str = "hist"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, density=True, alpha=0.7, edgecolor="black")
    ax.set_title("Нормализованная гистограмма")
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    save_figure(fig, f"task15_{name}")
    plt.close(fig)
    return fig


# ============ Задание 16. Отсортированные столбцы ============
def show_sorted(df: pd.DataFrame) -> None:
    for col in df.columns:
        print(f"\n--- {col} (отсортировано) ---")
        print(np.sort(df[col].to_numpy()))


# ============ Задание 17. Эмпирическая функция распределения ============
def plot_ecdf(x: np.ndarray, name: str = "ecdf"):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots()
    ax.step(xs, ys, where="post")
    ax.set_title("Эмпирическая функция распределения")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    save_figure(fig, f"task17_{name}")
    plt.close(fig)
    return fig


# ============ Задание 18. Статистики ============
def column_stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=0)),
        "mode": float(stats.mode(x, keepdims=False).mode),
        "median": float(np.median(x)),
    }


def print_stats(data: np.ndarray, labels: list) -> None:
    print("\n=== Статистики столбцов ===")
    for i, col in enumerate(labels):
        s = column_stats(data[:, i])
        print(f"\n[{col}]")
        for k, v in s.items():
            print(f"  {k:8s} = {v:.6f}")


# ============ Задание 19. Доверительные интервалы ============
def ci_mean(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    se = s / np.sqrt(n)
    return stats.t.interval(1 - alpha, df=n - 1, loc=m, scale=se)


def ci_var(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    s2 = np.var(x, ddof=1)
    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    return (n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low


def print_ci(data: np.ndarray, labels: list, alpha: float = 0.05) -> None:
    print(f"\n=== Доверительные интервалы (α = {alpha}) ===")
    for i, col in enumerate(labels):
        cm = ci_mean(data[:, i], alpha)
        cv = ci_var(data[:, i], alpha)
        print(f"\n[{col}]")
        print(f"  Среднее:    {cm}")
        print(f"  Дисперсия:  {cv}")


# ============ Задание 20. Ковариация и корреляция ============
def print_cov_corr(data: np.ndarray, labels: list) -> None:
    cov = np.cov(data, rowvar=False)
    corr = np.corrcoef(data, rowvar=False)
    print("\n=== Ковариационная матрица ===")
    print(pd.DataFrame(cov, index=labels, columns=labels))
    print("\n=== Корреляционная матрица ===")
    print(pd.DataFrame(corr, index=labels, columns=labels))


# ============ Задание 21. Значимость корреляции ============
def print_corr_significance(x: np.ndarray, y: np.ndarray, names=("x", "y")) -> None:
    r, p = stats.pearsonr(x, y)
    print(f"\n=== Проверка значимости корреляции {names[0]}-{names[1]} ===")
    print(f"  r = {r:.4f}, p-value = {p:.4f}")
    if p < 0.05:
        print("  → корреляция ЗНАЧИМА")
    else:
        print("  → корреляция НЕ значима")


# ============ Задание 22. Взаимная корреляция ============
def cross_correlation(x, y, max_lags: int = 50):
    lags = np.arange(-max_lags, max_lags + 1)
    c = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    c = c / (np.std(x) * np.std(y) * len(x))
    mid = len(c) // 2
    return lags, c[mid - max_lags: mid + max_lags + 1]


# ============ Задание 23. Производная и градиент ============
def compute_gradients(data: np.ndarray) -> np.ndarray:
    return np.gradient(data, axis=0)


# ============ Задание 24. Свёртка ============
def convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.convolve(x, y, mode="full")


# ============ Задание 25. Скалярное и векторное произведения ============
def dot_product(x, y):
    return np.dot(x, y)


def cross_product(x, y):
    return np.cross(x[:3], y[:3])


# ============ Задание 26. Нормы L1 и L2 ============
def norms(x: np.ndarray) -> dict:
    return {"L1": float(np.linalg.norm(x, ord=1)),
            "L2": float(np.linalg.norm(x, ord=2))}


# ============ Задание 27. Проверка гипотез ============
# ============ Задание 27. Проверка гипотез ============
def test_uniform(x: np.ndarray, alpha: float = 0.05):
    """Проверка гипотезы о равномерном распределении (совместимо со всеми SciPy)."""
    from scipy.stats import uniform

    a = float(np.min(x))
    b = float(np.max(x))

    def cdf_uniform(v):
        return uniform.cdf(v, loc=a, scale=b - a)

    stat, p = stats.kstest(x, cdf_uniform)

    print(f"\n[Гипотеза о равномерном распределении]")
    print(f"  KS = {stat:.4f}, p = {p:.4f}")
    print("  → отвергаем" if p < alpha else "  → не отвергаем")
    return stat, p


def test_normal(x: np.ndarray, alpha: float = 0.05):
    """Проверка гипотезы о нормальном распределении (совместимо со всеми SciPy)."""
    from scipy.stats import norm

    mean_ = float(np.mean(x))
    std_ = float(np.std(x, ddof=1))

    def cdf_normal(v):
        return norm.cdf(v, loc=mean_, scale=std_)

    stat, p = stats.kstest(x, cdf_normal)

    # Дополнительно — тест Шапиро-Уилка (мощнее)
    sh_stat, sh_p = stats.shapiro(x)

    print(f"\n[Гипотеза о нормальном распределении]")
    print(f"  KS:      stat = {stat:.4f}, p = {p:.4f}")
    print(f"  Shapiro: stat = {sh_stat:.4f}, p = {sh_p:.4f}")
    print("  → отвергаем" if sh_p < alpha else "  → не отвергаем")
    return stat, p


# ============ Задания 28–30. Спектрограмма, периодограмма, АЧХ ============
def plot_spectrogram(x: np.ndarray, fs: float = 1.0, name: str = "spec"):
    f, t, Sxx = spectrogram(x, fs=fs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    ax.set_ylabel("Частота")
    ax.set_xlabel("Время")
    ax.set_title("Спектрограмма")
    save_figure(fig, f"task28_{name}")
    plt.close(fig)


def plot_periodogram(x: np.ndarray, fs: float = 1.0, name: str = "per"):
    f, Pxx = periodogram(x, fs=fs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f, Pxx)
    ax.set_xlabel("Частота")
    ax.set_ylabel("PSD")
    ax.set_title("Периодограмма")
    save_figure(fig, f"task29_{name}")
    plt.close(fig)


def plot_fft_amplitude(x: np.ndarray, name: str = "fft"):
    n = len(x)
    fft = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0)
    half = n // 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq[:half], np.abs(fft[:half]))
    ax.set_xlabel("Частота")
    ax.set_ylabel("|X(f)|")
    ax.set_title("АЧХ через FFT")
    save_figure(fig, f"task30_{name}")
    plt.close(fig)


# ============ Задания 31–32. Интерполяция ============
def cubic_interpolation(x: np.ndarray, factor: int = 10):
    idx = np.arange(len(x))
    f = interp1d(idx, x, kind="cubic", fill_value="extrapolate")
    x_new = np.linspace(0, len(x) - 1, factor * len(x))
    return x_new, f(x_new)


def spline_interpolation(x: np.ndarray, factor: int = 10):
    idx = np.arange(len(x))
    spl = UnivariateSpline(idx, x, k=3, s=0)
    x_new = np.linspace(0, len(x) - 1, factor * len(x))
    return x_new, spl(x_new)


def plot_interpolations(x: np.ndarray, name: str = "interp"):
    x_new, y_cubic = cubic_interpolation(x)
    _, y_spline = spline_interpolation(x)
    idx = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, x, "o", label="Исходные", markersize=3)
    ax.plot(x_new, y_cubic, "-", label="Кубическая", alpha=0.7)
    ax.plot(x_new, y_spline, "--", label="Сплайн", alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Интерполяция")
    save_figure(fig, f"task31_32_{name}")
    plt.close(fig)


# ============ Задание 33. Бинарные маски ============
def binary_masks(x: np.ndarray) -> dict:
    return {
        "positive": x > 0,
        "negative": x < 0,
        "zero": x == 0,
        "interval": (x >= -1) & (x <= 1),
    }


def print_masks_info(data: np.ndarray, labels: list) -> None:
    print("\n=== Бинарные маски ===")
    for i, col in enumerate(labels):
        m = binary_masks(data[:, i])
        print(f"\n[{col}]")
        for k, v in m.items():
            print(f"  {k:10s}: {int(v.sum())} элементов")
"""Задания 38–49: тензоры TensorFlow, PyTorch, Keras, SciPy, sklearn."""
import numpy as np
import tensorflow as tf
import torch
import keras
from keras import ops
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from src.preprocessing import MinMaxScalerCustom, StandardScalerCustom


# ============ Задания 38–39. Тензоры ============
def to_tensorflow(data: np.ndarray) -> tf.Tensor:
    return tf.constant(data, dtype=tf.float32)


def to_pytorch(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)


# ============ Задания 40–42. AXW + B ============
def run_axw_b_tensorflow(X: tf.Tensor, k: int = 5, p: int = 4):
    n, m = X.shape
    A = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
    W = tf.random.normal((m, p))
    B = tf.random.uniform((k, p))
    result = tf.matmul(tf.matmul(tf.cast(A, tf.float32), X), W) + B
    return A, W, B, result


def run_axw_b_pytorch(X: torch.Tensor, k: int = 5, p: int = 4):
    n, m = X.shape
    A = torch.randint(0, 10, (k, n), dtype=torch.float32)
    W = torch.randn(m, p)
    B = torch.rand(k, p)
    result = A @ X @ W + B
    return A, W, B, result


# ============ Задания 43–45. Операция V = |sin(T) - exp(P)*sqrt(Q)| ============
def generate_TPQ():
    T = np.random.rand(3, 10)
    P = np.random.rand(3, 10)
    Q = np.random.rand(3, 10)
    return T, P, Q


def compute_V_keras(T, P, Q):
    T_k = ops.convert_to_tensor(T)
    P_k = ops.convert_to_tensor(P)
    Q_k = ops.convert_to_tensor(Q)
    return ops.abs(ops.sin(T_k) - ops.exp(P_k) * ops.sqrt(Q_k))


def compute_V_pytorch(T, P, Q):
    T_t = torch.tensor(T)
    P_t = torch.tensor(P)
    Q_t = torch.tensor(Q)
    return torch.abs(torch.sin(T_t) - torch.exp(P_t) * torch.sqrt(Q_t))


# ============ Задание 46. Разреженная матрица ============
def sparse_matrix(rows: int = 100_000, cols: int = 100_000, density: float = 1e-5):
    S = sparse.random(rows, cols, density=density, format="csr")
    print(f"[sparse] Форма: {S.shape}, NNZ = {S.nnz}")
    return S


# ============ Задание 47. Обратная матрица ============
def inverse_matrix(size: int = 100):
    M = np.random.rand(size, size)
    det = np.linalg.det(M)
    print(f"[inv] det = {det:.4e}")
    if abs(det) > 1e-10:
        M_inv = np.linalg.inv(M)
        ok = np.allclose(M @ M_inv, np.eye(size), atol=1e-6)
        print(f"[inv] Проверка M @ M_inv ≈ I: {ok}")
        return M_inv
    print("[inv] Матрица вырождена")
    return None


# ============ Задание 48. Достроить массив до F ============
def build_F(data: np.ndarray) -> np.ndarray:
    X_mm = MinMaxScalerCustom().fit_transform(data)
    X_std = StandardScalerCustom().fit_transform(data)
    F = np.stack([data, X_mm, X_std], axis=-1)
    print(f"[F] Форма тензора F: {F.shape}")
    return F


# ============ Задание 49. Пайплайн sklearn (PCA) ============
def run_pca_pipeline(F: np.ndarray):
    """Задание 49. Пайплайн sklearn: scaler + PCA.

    Вместо score_samples (который падает на вырожденной матрице)
    используем explained_variance_ratio_.sum() — суммарная доля
    объяснённой дисперсии. Больше — лучше.
    """
    import matplotlib.pyplot as plt
    from src.config import save_figure

    flat = F.reshape(len(F), -1)
    print(f"[PCA] Форма входа: {flat.shape}")

    scalers = {"minmax": MinMaxScaler(), "std": StandardScaler()}
    results = {}

    for name, scaler in scalers.items():
        pipe = Pipeline([("scaler", scaler), ("pca", PCA())])
        pipe.fit(flat)
        evr = pipe.named_steps["pca"].explained_variance_ratio_.sum()
        results[name] = evr
        print(f"[PCA] {name}: explained variance ratio = {evr:.4f}")

    best = max(results, key=results.get)
    print(f"[PCA] Лучший вариант: {best}")

    # Логарифмический график explained_variance_ratio_
    pca = PCA().fit(flat)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Кумулятивная доля дисперсии (log)")
    ax.set_title("PCA: explained variance ratio")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "task49_pca_evr")
    plt.close(fig)

    return results, best
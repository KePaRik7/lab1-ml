"""Задание 50: обработка изображения средствами SciPy."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from sklearn.decomposition import PCA
from skimage.color import rgb2gray

from src.config import save_figure


def split_channels(img):
    """1. Разделение на R, G, B каналы."""
    r = img[..., 0].copy()
    g = img[..., 1].copy()
    b = img[..., 2].copy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(r, cmap="Reds"); axes[1].set_title("Красный")
    axes[2].imshow(g, cmap="Greens"); axes[2].set_title("Зелёный")
    axes[3].imshow(b, cmap="Blues"); axes[3].set_title("Синий")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_channels")
    plt.close(fig)
    return {"r": r, "g": g, "b": b}


def rotate_image(img, angle=40.0):
    """2. Поворот на 40°."""
    rotated = rotate(img.astype(np.float32), angle=angle, reshape=True,
                     order=3, mode="constant")
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(rotated); axes[1].set_title(f"Поворот {angle}°")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_rotated")
    plt.close(fig)
    return rotated


def to_grayscale(img):
    """3. Преобразование в ч/б.

    Поддерживает RGB (3 канала) и RGBA (4 канала) — альфа-канал
    отбрасывается перед rgb2gray.
    """
    # Убираем альфа-канал, если он есть
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    # Приводим к [0, 1] для skimage.rgb2gray
    if img.max() > 1.0:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)

    gray = rgb2gray(img_float) if img_float.ndim == 3 else img_float
    gray_u8 = (gray * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(gray_u8, cmap="gray"); axes[1].set_title("Ч/Б")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_gray")
    plt.close(fig)
    return gray_u8

def gray_histogram(gray, bins=256):
    """4. Гистограмма ч/б изображения."""
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], hist, width=1, color="black", align="edge")
    ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Частота")
    ax.set_xlim(0, 255)
    save_figure(fig, "task50_gray_hist")
    plt.close(fig)
    return {"hist": hist, "bin_edges": bin_edges}


def split_by_histogram(gray, hist_info):
    """5. Разбиение на 3 части по гистограмме."""
    hist = hist_info["hist"]
    edges = hist_info["bin_edges"]
    cum = np.cumsum(hist)
    total = cum[-1]
    t1 = edges[np.searchsorted(cum, total * 0.33)]
    t2 = edges[np.searchsorted(cum, total * 0.66)]

    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(part1, cmap="gray"); axes[0].set_title(f"Тёмные (≤ {t1})")
    axes[1].imshow(part2, cmap="gray"); axes[1].set_title(f"Средние ({t1}-{t2})")
    axes[2].imshow(part3, cmap="gray"); axes[2].set_title(f"Светлые (> {t2})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_split3")
    plt.close(fig)
    return part1, part2, part3


def add_circular_frame(img):
    """6. Чёрная круговая рамка."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    R = min(h, w) / 2.0 - 2
    Y, X = np.ogrid[:h, :w]
    mask_outside = (X - cx) ** 2 + (Y - cy) ** 2 > R ** 2

    framed = img.copy()
    if framed.ndim == 2:
        framed[mask_outside] = 0
    else:
        framed[mask_outside, :] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(framed); axes[1].set_title("В круговой рамке")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_framed")
    plt.close(fig)
    return framed


def add_noise(img, sigma=25.0):
    """7. Случайный гауссовский шум."""
    noise = np.random.normal(0.0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(noisy); axes[1].set_title(f"Шум (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_noisy")
    plt.close(fig)
    return noisy


def gaussian_blur(img, sigma=2.0):
    """8. Гауссовское размытие."""
    if img.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
             for c in range(img.shape[2])], axis=-1)
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(blurred); axes[1].set_title(f"Размытие (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_blurred")
    plt.close(fig)
    return blurred


def sharpen_image(img, sigma=2.0, amount=1.5):
    """9. Sharpening (unsharp masking)."""
    if img.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
             for c in range(img.shape[2])], axis=-1)
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)

    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(sharpened); axes[1].set_title(f"Sharpening (amount={amount})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_sharpened")
    plt.close(fig)
    return sharpened


def apply_pca_to_image(img, n_components=32):
    """10. PCA к изображению."""
    h, w = img.shape[:2]

    if img.ndim == 3:
        flat = img.reshape(-1, img.shape[2]).astype(np.float32)
        features_per_pixel = img.shape[2]
    else:
        flat = img.reshape(-1, 1).astype(np.float32)
        features_per_pixel = 1

    n_comp = min(n_components, flat.shape[1] * 4) if features_per_pixel == 1 else n_components
    if features_per_pixel == 3:
        n_comp = min(n_components, 3)

    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(flat)
    restored = pca.inverse_transform(transformed)
    restored_img = restored.reshape(h, w, features_per_pixel).astype(np.float32)
    restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

    if features_per_pixel == 1:
        restored_img = restored_img[..., 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
    axes[0].set_title("Исходное")
    axes[1].imshow(restored_img, cmap="gray" if img.ndim == 2 else None)
    axes[1].set_title(f"PCA ({n_comp} компонент)")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_pca_restored")
    plt.close(fig)

    return {
        "restored": restored_img,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def run_task50(image_path: str) -> dict:
    """Последовательно выполняет все 10 подзаданий задания 50."""
    img = plt.imread(image_path)

    # Если картинка RGBA — отбрасываем альфа-канал
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    # Приводим к uint8 [0, 255]
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    print(f"[task50] Изображение: shape={img.shape}, dtype={img.dtype}")

    results = {}
    print("[task50] 1. Разделение каналов...")
    results["channels"] = split_channels(img)

    print("[task50] 2. Поворот на 40°...")
    results["rotated"] = rotate_image(img, angle=40.0)

    print("[task50] 3. Ч/Б...")
    results["gray"] = to_grayscale(img)

    print("[task50] 4. Гистограмма...")
    results["hist"] = gray_histogram(results["gray"])

    print("[task50] 5. Разбиение по гистограмме...")
    results["parts"] = split_by_histogram(results["gray"], results["hist"])

    print("[task50] 6. Круговая рамка...")
    results["framed"] = add_circular_frame(img)

    print("[task50] 7. Шум...")
    results["noisy"] = add_noise(img, sigma=25.0)

    print("[task50] 8. Размытие...")
    results["blurred"] = gaussian_blur(img, sigma=2.0)

    print("[task50] 9. Sharpening...")
    results["sharpened"] = sharpen_image(img, sigma=2.0, amount=1.5)

    print("[task50] 10. PCA...")
    results["pca"] = apply_pca_to_image(img, n_components=32)

    print("[task50] Все графики сохранены в output/")
    return results
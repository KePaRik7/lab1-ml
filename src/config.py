"""Конфигурация проекта: seed, пути, вспомогательные функции."""
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
import torch

# Директории
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seeds(seed: int = 42) -> None:
    """Фиксирует seed для всех генераторов случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[config] Seed зафиксирован: {seed}")


def timestamped_filename(base: str, ext: str) -> str:
    """Возвращает путь output/<base>_<timestamp>.<ext>."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_DIR, f"{base}_{ts}.{ext}")


def save_figure(fig, base_name: str) -> str:
    """Сохраняет фигуру Matplotlib в PNG с timestamp в имени."""
    path = timestamped_filename(base_name, "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[config] Рисунок сохранён: {path}")
    return path
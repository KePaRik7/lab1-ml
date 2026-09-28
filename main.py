"""Главный скрипт лабораторной работы №1. Вариант 5 (столбцы t, z).

Файл датасета имеет многоуровневую шапку:
  строка 0  -> 'Вариант 1', 'Вариант 2', ..., 'Вариант 24'
  строка 1  -> 'x', 'y', 't', 'u', 'x', 'z', ...
  строка 2+ -> числовые данные

Вариант 5 соответствует столбцам I и J (индексы 8 и 9).
Их подзаголовки в вашем файле: 't' и 'z'.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import set_seeds, DATA_DIR, save_figure
from src.io_utils import (
    read_table, write_table, check_data_quality, print_quality_report,
    show_table, extract_columns, cast_types,
)
from src.preprocessing import (
    auto_clean, to_numpy, MinMaxScalerCustom, StandardScalerCustom,
    split_train_val_test, sliding_window, moving_average,
)
from src.analysis import (
    plot_series, plot_histogram, show_sorted, plot_ecdf,
    print_stats, print_ci, print_cov_corr, print_corr_significance,
    cross_correlation, compute_gradients, convolution,
    dot_product, cross_product, norms,
    test_uniform, test_normal,
    plot_spectrogram, plot_periodogram, plot_fft_amplitude,
    plot_interpolations, binary_masks, print_masks_info,
)
from src.tensors import (
    to_tensorflow, to_pytorch, run_axw_b_tensorflow, run_axw_b_pytorch,
    generate_TPQ, compute_V_keras, compute_V_pytorch,
    sparse_matrix, inverse_matrix, build_F, run_pca_pipeline,
)

# ---- Параметры варианта 5 ----
# В датасете Вариант 5 = столбцы I и J = индексы 8 и 9
# Их подзаголовки: 't' и 'z'
VARIANT_COL_INDICES = [8, 9]
VARIANT_COLUMNS = ["t", "z"]


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Читает Excel с двухуровневой шапкой, возвращает таблицу данных.

    Делает имена столбцов уникальными (добавляет _индекс), чтобы избежать
    проблем с дубликатами (x, y, t, u встречаются по несколько раз).
    """
    df_raw = pd.read_excel(dataset_path, header=None)
    print(f"[Задание 1] Сырой файл: {df_raw.shape}")

    # Строка 0 — 'Вариант 1' ... 'Вариант 24'
    # Строка 1 — подзаголовки 'x', 'y', 't', 'u', ...
    # Строка 2+ — данные
    sub_headers = df_raw.iloc[1].tolist()
    data_rows = df_raw.iloc[2:].reset_index(drop=True)

    # Уникальные имена: x_0, y_1, t_2, u_3, x_4, z_5, col_6, y_7, t_8, z_9, ...
    unique_cols = []
    for i, s in enumerate(sub_headers):
        if pd.notna(s) and str(s).strip():
            unique_cols.append(f"{str(s).strip()}_{i}")
        else:
            unique_cols.append(f"col_{i}")
    data_rows.columns = unique_cols

    # Отрезаем полностью пустые строки
    before = len(data_rows)
    data_rows = data_rows.dropna(how="all").reset_index(drop=True)
    after = len(data_rows)
    if before != after:
        print(f"[Задание 1] Удалено полностью пустых строк: {before - after}")

    print(f"[Задание 1] Данные после разбора шапки: {data_rows.shape}")
    print(f"[Задание 1] Имена столбцов (первые 20): {list(data_rows.columns)[:20]}")
    return data_rows


def main():
    # Задание 13: фиксация seed для всех генераторов
    set_seeds(42)

    # =========================================================
    # РАЗДЕЛ 1. Считывание и запись данных
    # =========================================================
    print("\n" + "=" * 60)
    print("РАЗДЕЛ 1. Считывание и запись данных")
    print("=" * 60)

    # ---------- Задание 1: чтение файла ----------
    dataset_path = os.path.join(DATA_DIR, "Lab1_Dataset.xlsx")
    df_full = load_dataset(dataset_path)

    # ---------- Задание 4: проверка качества ----------
    print("\n[Задание 4] Проверка качества данных...")
    print_quality_report(df_full)

    # ---------- Задание 5: вывод таблицы ----------
    show_table(df_full, n=5)

    # ---------- Задание 6: извлечение столбцов варианта ----------
    print(f"\n[Задание 6] Извлечение столбцов Варианта 5 "
          f"(индексы {VARIANT_COL_INDICES})...")
    df_var = df_full.iloc[:, VARIANT_COL_INDICES].copy()
    df_var.columns = VARIANT_COLUMNS
    print(f"  Форма извлечённой таблицы: {df_var.shape}")
    print(f"  Заголовки: {list(df_var.columns)}")
    print(df_var.head())

    # ---------- Задание 7: явные типы (безопасное приведение) ----------
    print("\n[Задание 7] Приведение типов к float64...")
    for col in df_var.columns:
        df_var[col] = pd.to_numeric(df_var[col], errors="coerce")
    print(f"  Типы: {dict(df_var.dtypes)}")
    print(f"  Пропусков после приведения: {df_var.isna().sum().to_dict()}")

    # ---------- Задание 2: запись во все форматы ----------
    print("\n[Задание 2] Сохранение извлечённой таблицы во все форматы...")
    write_table(df_var, "variant5_raw")

    # =========================================================
    # РАЗДЕЛ 2. Предобработка данных
    # =========================================================
    print("\n" + "=" * 60)
    print("РАЗДЕЛ 2. Предобработка данных")
    print("=" * 60)

    # ---------- Задание 8: обработка пропусков ----------
    print("\n[Задание 8] Обработка пропусков (mean-заполнение)...")
    df_clean = auto_clean(df_var, list(df_var.columns))
    print(f"  Пропусков после очистки: {df_clean.isna().sum().to_dict()}")

    # ---------- Задание 9: DataFrame -> NumPy ----------
    data = to_numpy(df_clean)
    print(f"\n[Задание 9] NumPy-массив: форма {data.shape}, dtype {data.dtype}")

    # ---------- Задание 10: масштабирование ----------
    print("\n[Задание 10] Масштабирование...")
    mm = MinMaxScalerCustom(a=0.0, b=1.0).fit(data)
    data_mm = mm.transform(data)

    mm_ab = MinMaxScalerCustom(a=-1.0, b=1.0).fit(data)
    data_mm_ab = mm_ab.transform(data)

    std = StandardScalerCustom().fit(data)
    data_std = std.transform(data)

    print(f"  MinMax [0, 1]  : min={data_mm.min():.4f}, max={data_mm.max():.4f}")
    print(f"  MinMax [-1, 1] : min={data_mm_ab.min():.4f}, max={data_mm_ab.max():.4f}")
    print(f"  Standard       : mean={data_std.mean():.4f}, std={data_std.std():.4f}")

    # ---------- Задание 11: инверсия ----------
    print("\n[Задание 11] Проверка инверсии масштабирования...")
    assert np.allclose(mm.inverse_transform(data_mm), data, atol=1e-9)
    assert np.allclose(mm_ab.inverse_transform(data_mm_ab), data, atol=1e-9)
    assert np.allclose(std.inverse_transform(data_std), data, atol=1e-9)
    print("  Инверсия корректна для всех трёх методов.")

    # ---------- Задание 12: разбиение ----------
    print("\n[Задание 12] Разбиение на train/val/test (70/15/15)...")
    x_tr, x_val, x_te = split_train_val_test(data, (70, 15, 15), percent=True)
    print(f"  train: {x_tr.shape}, val: {x_val.shape}, test: {x_te.shape}")

    # =========================================================
    # РАЗДЕЛ 3. Работа с данными
    # =========================================================
    print("\n" + "=" * 60)
    print("РАЗДЕЛ 3. Работа с данными")
    print("=" * 60)

    # ---------- Задание 14 ----------
    print("\n[Задание 14] График исходных данных...")
    plot_series(data, title="Вариант 5: столбцы t и z")

    # ---------- Задание 15 ----------
    print("[Задание 15] Гистограммы по каждому столбцу...")
    for i, col in enumerate(VARIANT_COLUMNS):
        plot_histogram(data[:, i], bins=30, name=col)

    # ---------- Задание 16 ----------
    print("\n[Задание 16] Отсортированные по возрастанию столбцы:")
    show_sorted(df_clean)

    # ---------- Задание 17 ----------
    print("\n[Задание 17] Эмпирические функции распределения...")
    for i, col in enumerate(VARIANT_COLUMNS):
        plot_ecdf(data[:, i], name=col)

    # ---------- Задание 18 ----------
    print_stats(data, VARIANT_COLUMNS)

    # ---------- Задание 19 ----------
    print_ci(data, VARIANT_COLUMNS, alpha=0.05)

    # ---------- Задание 20 ----------
    print_cov_corr(data, VARIANT_COLUMNS)

    # ---------- Задание 21 ----------
    print_corr_significance(data[:, 0], data[:, 1], names=tuple(VARIANT_COLUMNS))

    # ---------- Задание 22 ----------
    print("\n[Задание 22] Взаимная корреляция...")
    lags, cc = cross_correlation(data[:, 0], data[:, 1], max_lags=50)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(lags, cc)
    ax.set_title(f"Взаимная корреляция столбцов {VARIANT_COLUMNS[0]} и {VARIANT_COLUMNS[1]}")
    ax.set_xlabel("Лаг")
    ax.set_ylabel("Корреляция")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "task22_cross_corr")
    plt.close(fig)

    # ---------- Задание 23 ----------
    print("\n[Задание 23] Производная и градиент...")
    grad = compute_gradients(data)
    print(f"  Форма градиента: {grad.shape}")

    # ---------- Задание 24 ----------
    print("\n[Задание 24] Свёртка столбцов...")
    conv = convolution(data[:, 0], data[:, 1])
    print(f"  Длина результата свёртки: {len(conv)}")

    # ---------- Задание 25 ----------
    print("\n[Задание 25] Произведения:")
    print(f"  Скалярное {VARIANT_COLUMNS[0]}·{VARIANT_COLUMNS[1]}        = "
          f"{dot_product(data[:, 0], data[:, 1]):.4f}")
    print(f"  Векторное (первые 3)                   = "
          f"{cross_product(data[:, 0], data[:, 1])}")

    # ---------- Задание 26 ----------
    print("\n[Задание 26] Нормы L1 и L2:")
    for i, col in enumerate(VARIANT_COLUMNS):
        n = norms(data[:, i])
        print(f"  [{col}] L1 = {n['L1']:.4f}, L2 = {n['L2']:.4f}")

    # ---------- Задание 27 ----------
    print("\n[Задание 27] Проверка гипотез о распределении:")
    test_uniform(data[:, 0])   # 1-й столбец — равномерное
    test_normal(data[:, 1])    # 2-й столбец — нормальное

    # ---------- Задания 28–30 ----------
    print("\n[Задания 28–30] Спектрограмма / периодограмма / АЧХ...")
    for i, col in enumerate(VARIANT_COLUMNS):
        plot_spectrogram(data[:, i], name=col)
        plot_periodogram(data[:, i], name=col)
        plot_fft_amplitude(data[:, i], name=col)

    # ---------- Задания 31–32 ----------
    print("\n[Задания 31–32] Кубическая интерполяция и сплайны...")
    for i, col in enumerate(VARIANT_COLUMNS):
        plot_interpolations(data[:, i], name=col)

    # ---------- Задание 33 ----------
    print_masks_info(data, VARIANT_COLUMNS)

    # ---------- Задания 34–35 ----------
    print("\n[Задания 34–35] Сравнение с sklearn...")
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    x_col = data[:, [0]]
    sk_mm = MinMaxScaler().fit(x_col)
    sk_std = StandardScaler().fit(x_col)

    assert np.allclose(mm.transform(x_col)[:, [0]], sk_mm.transform(x_col), atol=1e-9)
    assert np.allclose(std.transform(x_col)[:, [0]], sk_std.transform(x_col), atol=1e-9)
    print("  Совпадение собственных реализаций и sklearn подтверждено (atol=1e-9).")

    # ---------- Задания 36–37 ----------
    print("\n[Задания 36–37] Скользящее окно и среднее (width=5)...")
    w = sliding_window(data[:, 0], 5)
    ma = moving_average(data[:, 0], 5)
    print(f"  Форма матрицы окон: {w.shape}")
    print(f"  Длина скользящего среднего: {len(ma)}")

    # =========================================================
    # Тензорные вычисления (задания 38–49)
    # =========================================================
    print("\n" + "=" * 60)
    print("Тензорные вычисления (задания 38–49)")
    print("=" * 60)

    X_tf = to_tensorflow(data)
    X_pt = to_pytorch(data)
    print(f"[38] TensorFlow тензор: shape = {X_tf.shape}, dtype = {X_tf.dtype}")
    print(f"[39] PyTorch тензор   : shape = {tuple(X_pt.shape)}, dtype = {X_pt.dtype}")

    A, W, B, res_tf = run_axw_b_tensorflow(X_tf, k=5, p=4)
    print(f"[41] TF: A({A.shape}) @ X({X_tf.shape}) @ W({W.shape}) + B({B.shape}) "
          f"-> {res_tf.shape}")

    A_pt, W_pt, B_pt, res_pt = run_axw_b_pytorch(X_pt, k=5, p=4)
    print(f"[42] PT: A({tuple(A_pt.shape)}) @ X({tuple(X_pt.shape)}) @ "
          f"W({tuple(W_pt.shape)}) + B({tuple(B_pt.shape)}) -> {tuple(res_pt.shape)}")

    T, P, Q = generate_TPQ()
    V_keras = compute_V_keras(T, P, Q)
    V_torch = compute_V_pytorch(T, P, Q)
    print(f"[44] Keras V: shape = {V_keras.shape}")
    print(f"[45] Torch V: shape = {tuple(V_torch.shape)}")

    #print("\n[Задание 46] Разреженная матрица 100 000 × 100 000...")
    #S = sparse_matrix(100_000, 100_000, 1e-5)

    print("\n[Задание 47] Обратная матрица 100×100...")
    inverse_matrix(100)

    print("\n[Задание 48] Тензор F (исходные + MinMax + Standard)...")
    F = build_F(data)

    print("\n[Задание 49] Пайплайн sklearn: scaler + PCA...")
    run_pca_pipeline(F)

    # =========================================================
    # Задание 50
    # =========================================================
    print("\n" + "=" * 60)
    print("Задание 50. Обработка изображения средствами SciPy")
    print("=" * 60)
    image_path = os.path.join(DATA_DIR, "image.png")
    if os.path.exists(image_path):
        from src.image_proc import run_task50
        run_task50(image_path)
    else:
        print(f"[task50] Файл {image_path} не найден — задание пропущено.")
        print("       Положите любое изображение image.png в папку data/.")

    # ---------- Финальное сохранение ----------
    print("\n[Сохранение] Запись очищенной таблицы во все форматы...")
    write_table(df_clean, "variant5_clean")

    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print(f"Все результаты сохранены в папке: {os.path.abspath('output')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
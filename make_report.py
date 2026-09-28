"""Генерация отчёта по лабораторной работе №1 в формате .docx.

Собирает титульный лист, разделы по заданиям, автоматически
вставляет картинки из папки output/ и добавляет подписи.
"""
import os
import glob
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUTPUT_DIR = "output"
REPORT_PATH = "Отчет_Лаб1_Вариант5.docx"


# ---------- Вспомогательные функции ----------
def set_style(doc):
    """Настраивает базовый стиль документа."""
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    # Кириллица
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    return p


def add_paragraph(doc, text, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    if align is not None:
        p.alignment = align
    return p


def add_code(doc, code: str):
    """Моноширинный блок кода."""
    p = doc.add_paragraph()
    run = p.add_run(code)
    run.font.name = "Consolas"
    run.font.size = Pt(10)
    return p


def add_image_with_caption(doc, image_path: str, caption: str, width_cm=14.0):
    """Вставляет картинку и подпись под ней."""
    if not os.path.exists(image_path):
        add_paragraph(doc, f"[Нет файла: {image_path}]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(image_path, width=Cm(width_cm))

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"
    cap.paragraph_format.space_after = Pt(12)


def find_one(pattern: str):
    """Находит первый файл по шаблону (из-за timestamp в имени)."""
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
    return files[0] if files else None


def add_table(doc, rows: list):
    """Таблица с данными."""
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(11)
    return table


# ---------- Основной скрипт ----------
def main():
    doc = Document()
    set_style(doc)

    # Поля страницы
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(3)
        section.right_margin = Cm(1.5)

    # ==================== ТИТУЛЬНЫЙ ЛИСТ ====================
    add_paragraph(doc, "МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ",
                  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "РОССИЙСКОЙ ФЕДЕРАЦИИ",
                  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "")
    add_paragraph(doc, "ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ",
                  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ",
                  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "«ДОНСКОЙ ГОСУДАРСТВЕННЫЙ ТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ»",
                  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "(ДГТУ)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "")
    add_paragraph(doc, "Кафедра «ИПМ Ростсельмаш»",
                  align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "")
    for _ in range(6):
        add_paragraph(doc, "")

    add_paragraph(doc, "ОТЧЁТ", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "по лабораторной работе №1",
                  align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "«Библиотеки Python для машинного обучения.",
                  align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "Работа с данными»",
                  align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "")
    add_paragraph(doc, "Вариант 5", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(6):
        add_paragraph(doc, "")

    add_paragraph(doc, "Выполнил: студент группы ИПБП41",
                  align=WD_ALIGN_PARAGRAPH.RIGHT)
    add_paragraph(doc, "_________________ / ____________ /",
                  align=WD_ALIGN_PARAGRAPH.RIGHT)
    add_paragraph(doc, "")
    add_paragraph(doc, "Проверил: _____________________",
                  align=WD_ALIGN_PARAGRAPH.RIGHT)
    for _ in range(4):
        add_paragraph(doc, "")
    add_paragraph(doc, "Ростов-на-Дону, 2026",
                  align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_page_break()

    # ==================== ВВЕДЕНИЕ ====================
    add_heading(doc, "Введение", level=1)
    add_paragraph(doc, "Цель работы: изучить основные библиотеки Python для "
                       "машинного обучения (NumPy, Pandas, Matplotlib, SciPy, "
                       "scikit-learn, TensorFlow, Keras, PyTorch) и приобрести "
                       "практические навыки работы с табличными данными: "
                       "чтение/запись, предобработка, статистический анализ, "
                       "тензорные вычисления, обработка изображений.")
    add_paragraph(doc, "Задачи:")
    for task in [
        "1. Освоить чтение и запись данных в различных форматах.",
        "2. Научиться выявлять и обрабатывать проблемы в данных.",
        "3. Реализовать собственные функции масштабирования и сравнить их со sklearn.",
        "4. Освоить статистический анализ данных средствами NumPy, Pandas, SciPy.",
        "5. Научиться работать с тензорами TensorFlow, PyTorch, Keras.",
        "6. Освоить базовую обработку изображений средствами SciPy.",
    ]:
        add_paragraph(doc, task)
    add_paragraph(doc, "")
    add_paragraph(doc, "Вариант 5: столбцы t и z (столбцы I и J исходного датасета).")

    doc.add_page_break()

    # ==================== РАЗДЕЛ 1 ====================
    add_heading(doc, "Раздел 1. Считывание и запись данных", level=1)

    add_heading(doc, "Задание 1. Чтение файлов", level=2)
    add_paragraph(doc, "Реализована универсальная функция read_table, "
                       "поддерживающая форматы .xlsx, .csv, .txt, .mat. "
                       "Для .mat используется scipy.io.loadmat, для .txt — "
                       "разделитель из пробелов и табуляций, для .csv и .xlsx — "
                       "стандартные функции pandas. При отсутствии заголовков "
                       "автоматически присваиваются имена col_0, col_1, …")
    add_paragraph(doc, "Датасет имеет двухуровневую шапку: строка 0 — "
                       "«Вариант 1» … «Вариант 24», строка 1 — подзаголовки "
                       "(x, y, t, u, …). Данные начинаются со строки 2. "
                       "Итоговая форма данных: 198 × 48.")

    add_heading(doc, "Задание 2. Запись файлов", level=2)
    add_paragraph(doc, "Реализована функция write_table, сохраняющая DataFrame "
                       "во все четыре формата с timestamp в имени. Пример "
                       "сохранённых файлов:")
    add_code(doc, "variant5_raw_2026-09-28_18-42-53.xlsx\n"
                  "variant5_raw_2026-09-28_18-42-53.csv\n"
                  "variant5_raw_2026-09-28_18-42-53.txt\n"
                  "variant5_raw_2026-09-28_18-42-53.mat")

    add_heading(doc, "Задание 3. Сохранение изображений", level=2)
    add_paragraph(doc, "Реализована функция save_figure, сохраняющая фигуры "
                       "Matplotlib в PNG с timestamp в имени. Все графики "
                       "в отчёте сохранены в папке output/.")

    add_heading(doc, "Задание 4. Проверка данных", level=2)
    add_paragraph(doc, "Реализована функция check_data_quality, проверяющая "
                       "пустые ячейки, некорректные данные и типы. Обнаружено:")
    add_paragraph(doc, "• Пропуски (NaN) в столбцах t (32 шт.) и z (31 шт.) "
                       "после приведения типов.")
    add_paragraph(doc, "• Некорректные значения: 'ERROR', '-' в столбцах "
                       "x_4, z_5, t_8, t_14, u_15, b_23, x_24, b_35, col_42, t_44.")

    add_heading(doc, "Задание 5. Вывод таблицы", level=2)
    add_paragraph(doc, "Выведены первые 5 строк таблицы, размерность (198, 48) "
                       "и типы данных.")

    add_heading(doc, "Задание 6. Извлечение столбцов по варианту", level=2)
    add_paragraph(doc, "Для Варианта 5 выбраны столбцы с индексами 8 и 9 "
                       "(подзаголовки t и z). Форма извлечённой таблицы: (198, 2). "
                       "Исходный датасет не изменялся (использована копия через .copy()).")

    add_heading(doc, "Задание 7. Явные типы данных", level=2)
    add_paragraph(doc, "Для столбцов t и z установлен тип float64 через "
                       "pd.to_numeric(errors='coerce'). Это безопасно "
                       "преобразует некорректные строки ('ERROR', '-') в NaN.")

    doc.add_page_break()

    # ==================== РАЗДЕЛ 2 ====================
    add_heading(doc, "Раздел 2. Предобработка данных", level=1)

    add_heading(doc, "Задание 8. Обработка пропусков", level=2)
    add_paragraph(doc, "Реализовано меню обработки проблем: drop, cast, mean, "
                       "ffill, bfill, zero, value. Для варианта 5 применена "
                       "стратегия cast → mean (приведение типов с последующим "
                       "заполнением средним). После обработки пропусков: 0.")
    add_paragraph(doc, "Пример функции обработки:")
    add_code(doc,
        "def handle_missing(df, col, action, value=None):\n"
        "    if action == 'mean':\n"
        "        df[col] = df[col].fillna(df[col].mean())\n"
        "    elif action == 'cast':\n"
        "        df[col] = pd.to_numeric(df[col], errors='coerce')\n"
        "    ...")

    add_heading(doc, "Задание 9. DataFrame → NumPy", level=2)
    add_paragraph(doc, "Очищенный DataFrame преобразован в массив NumPy "
                       "формы (198, 2) с типом float64.")

    add_heading(doc, "Задание 10. Функции масштабирования", level=2)
    add_paragraph(doc, "Реализованы классы MinMaxScalerCustom и "
                       "StandardScalerCustom. Первый поддерживает произвольные "
                       "границы [a, b] (по умолчанию [0, 1]). Второй — "
                       "стандартизацию (z-score). Параметры масштабирования "
                       "(min, max, mean, std) сохраняются в объектах.")
    add_paragraph(doc, "Формулы:")
    add_paragraph(doc, "• MinMax: x_new = a + (x - min) · (b - a) / (max - min)")
    add_paragraph(doc, "• MinMax в границах [a, b]: та же формула с параметрами")
    add_paragraph(doc, "• Standard: z = (x - μ) / σ")
    add_paragraph(doc, "Результаты:")
    add_paragraph(doc, "• MinMax [0, 1]: min = 0.0000, max = 1.0000")
    add_paragraph(doc, "• MinMax [-1, 1]: min = -1.0000, max = 1.0000")
    add_paragraph(doc, "• Standard: mean = 0.0000, std = 1.0000")

    add_heading(doc, "Задание 11. Восстановление масштаба", level=2)
    add_paragraph(doc, "Реализованы методы inverse_transform для обоих классов. "
                       "Проверка через np.allclose с atol=1e-9 прошла успешно.")

    add_heading(doc, "Задание 12. Разбиение на 3 части", level=2)
    add_paragraph(doc, "Реализована функция split_train_val_test с поддержкой "
                       "процентов и долей. Пример разбиения 70%:15%:15% для "
                       "массива 198 элементов:")
    add_paragraph(doc, "• train: (139, 2)")
    add_paragraph(doc, "• val: (30, 2)")
    add_paragraph(doc, "• test: (29, 2)")

    add_heading(doc, "Задание 13. Фиксация seed", level=2)
    add_paragraph(doc, "В начале main.py вызывается set_seeds(42), который "
                       "устанавливает seed для random, NumPy, TensorFlow и "
                       "PyTorch. Это обеспечивает воспроизводимость результатов.")

    doc.add_page_break()

    # ==================== РАЗДЕЛ 3 ====================
    add_heading(doc, "Раздел 3. Работа с данными", level=1)

    add_heading(doc, "Задание 14. График исходных данных", level=2)
    img = find_one("task14_series_*.png")
    add_image_with_caption(doc, img, "Рис. 1. Исходные данные: столбцы t и z")

    add_heading(doc, "Задание 15. Гистограмма и плотность", level=2)
    add_paragraph(doc, "Нормализованные гистограммы для оценки плотности "
                       "распределения по каждому столбцу.")
    add_image_with_caption(doc, find_one("task15_t_*.png"),
                           "Рис. 2. Гистограмма столбца t")
    add_image_with_caption(doc, find_one("task15_z_*.png"),
                           "Рис. 3. Гистограмма столбца z")

    add_heading(doc, "Задание 16. Отсортированные столбцы", level=2)
    add_paragraph(doc, "Столбцы отсортированы по возрастанию. Для столбца t "
                       "характерен широкий диапазон (от -5344 до 17207), для z — "
                       "компактный интервал [-1, 1].")

    add_heading(doc, "Задание 17. Эмпирическая функция распределения", level=2)
    add_image_with_caption(doc, find_one("task17_t_*.png"),
                           "Рис. 4. ЭФР столбца t")
    add_image_with_caption(doc, find_one("task17_z_*.png"),
                           "Рис. 5. ЭФР столбца z")

    add_heading(doc, "Задание 18. Статистики", level=2)
    add_paragraph(doc, "Выборочные характеристики столбцов:")
    add_table(doc, [
        ["Параметр", "Столбец t", "Столбец z"],
        ["Среднее", "1853.76", "0.0396"],
        ["Дисперсия", "11 651 183.86", "0.4503"],
        ["Мода", "1853.76", "0.0396"],
        ["Медиана", "1197.97", "0.0396"],
    ])

    add_heading(doc, "Задание 19. Доверительные интервалы (α = 0.05)", level=2)
    add_table(doc, [
        ["Столбец", "Среднее (нижняя, верхняя)", "Дисперсия (нижняя, верхняя)"],
        ["t", "1374.16 … 2333.36", "9 702 683.59 … 14 416 238.29"],
        ["z", "-0.0546 … 0.1339", "0.3750 … 0.5572"],
    ])

    add_heading(doc, "Задание 20. Ковариация и корреляция", level=2)
    add_paragraph(doc, "Ковариационная матрица:")
    add_table(doc, [
        ["", "t", "z"],
        ["t", "1.171033e+07", "203.28"],
        ["z", "203.28", "0.4526"],
    ])
    add_paragraph(doc, "Корреляционная матрица:")
    add_table(doc, [
        ["", "t", "z"],
        ["t", "1.000000", "0.0883"],
        ["z", "0.0883", "1.000000"],
    ])

    add_heading(doc, "Задание 21. Значимость коэффициента корреляции", level=2)
    add_paragraph(doc, "Тест Пирсона для пары t-z: r = 0.0883, p-value = 0.2161.")
    add_paragraph(doc, "Так как p-value > 0.05, корреляция НЕ значима.")

    add_heading(doc, "Задание 22. Взаимная корреляция", level=2)
    add_image_with_caption(doc, find_one("task22_cross_corr_*.png"),
                           "Рис. 6. Взаимная корреляция столбцов t и z")

    add_heading(doc, "Задание 23. Производная и градиент", level=2)
    add_paragraph(doc, "Вычислен np.gradient по оси строк. Форма градиента: (198, 2).")

    add_heading(doc, "Задание 24. Свёртка", level=2)
    add_paragraph(doc, "Длина результата свёртки столбцов t и z: 395.")

    add_heading(doc, "Задание 25. Скалярное и векторное произведения", level=2)
    add_paragraph(doc, "Скалярное произведение t·z = 54594.86.")
    add_paragraph(doc, "Векторное произведение (первые 3 элемента): "
                       "[0.7819, 0.1909, 0.6173].")

    add_heading(doc, "Задание 26. Нормы L1 и L2", level=2)
    add_table(doc, [
        ["Столбец", "L1", "L2"],
        ["t", "473 081.41", "54 656.62"],
        ["z", "112.96", "9.46"],
    ])

    add_heading(doc, "Задание 27. Проверка гипотез о распределении", level=2)
    add_paragraph(doc, "Для столбца t (1-й столбец) проверена гипотеза о "
                       "равномерном распределении: KS = 0.4390, p = 0.0000 → "
                       "гипотеза ОТВЕРГАЕТСЯ.")
    add_paragraph(doc, "Для столбца z (2-й столбец) проверена гипотеза о "
                       "нормальном распределении: KS = 0.1046, p = 0.0243; "
                       "Shapiro-Wilk = 0.9144, p = 0.0000 → гипотеза ОТВЕРГАЕТСЯ.")

    add_heading(doc, "Задания 28–30. Спектрограмма, периодограмма, АЧХ", level=2)
    add_image_with_caption(doc, find_one("task28_t_*.png"),
                           "Рис. 7. Спектрограмма столбца t")
    add_image_with_caption(doc, find_one("task29_t_*.png"),
                           "Рис. 8. Периодограмма столбца t")
    add_image_with_caption(doc, find_one("task30_t_*.png"),
                           "Рис. 9. АЧХ (через FFT) столбца t")
    add_image_with_caption(doc, find_one("task28_z_*.png"),
                           "Рис. 10. Спектрограмма столбца z")
    add_image_with_caption(doc, find_one("task29_z_*.png"),
                           "Рис. 11. Периодограмма столбца z")
    add_image_with_caption(doc, find_one("task30_z_*.png"),
                           "Рис. 12. АЧХ (через FFT) столбца z")

    add_heading(doc, "Задания 31–32. Интерполяция", level=2)
    add_paragraph(doc, "Кубическая интерполяция (interp1d, kind='cubic') и "
                       "интерполяция сплайнами (UnivariateSpline, k=3, s=0) "
                       "с увеличением числа точек в 10 раз.")
    add_image_with_caption(doc, find_one("task31_32_t_*.png"),
                           "Рис. 13. Интерполяция столбца t")
    add_image_with_caption(doc, find_one("task31_32_z_*.png"),
                           "Рис. 14. Интерполяция столбца z")

    add_heading(doc, "Задание 33. Бинарные маски", level=2)
    add_table(doc, [
        ["Маска", "Столбец t", "Столбец z"],
        ["Положительные", "156", "117"],
        ["Отрицательные", "42", "81"],
        ["Нулевые", "0", "0"],
        ["В интервале [-1, 1]", "3", "198"],
    ])

    add_heading(doc, "Задания 34–35. Сравнение с sklearn", level=2)
    add_paragraph(doc, "Собственные реализации MinMaxScalerCustom и "
                       "StandardScalerCustom совпали с sklearn "
                       "(MinMaxScaler, StandardScaler) с точностью atol=1e-9. "
                       "Инверсия масштабирования также проверена.")

    add_heading(doc, "Задания 36–37. Скользящее окно и среднее", level=2)
    add_paragraph(doc, "Для окна шириной 5 получена матрица окон формы "
                       "(194, 5) и скользящее среднее длины 194.")

    doc.add_page_break()

    # ==================== ТЕНЗОРНЫЕ ВЫЧИСЛЕНИЯ ====================
    add_heading(doc, "Тензорные вычисления (задания 38–49)", level=1)

    add_heading(doc, "Задания 38–39. Тензоры TensorFlow и PyTorch", level=2)
    add_paragraph(doc, "TF-тензор: shape (198, 2), dtype float32.")
    add_paragraph(doc, "PT-тензор: shape (198, 2), dtype torch.float32.")

    add_heading(doc, "Задания 40–42. Операция AXW + B", level=2)
    add_paragraph(doc, "Размерности подобраны так, чтобы операция была "
                       "определена: A(5, 198) @ X(198, 2) @ W(2, 4) + B(5, 4) "
                       "= (5, 4). Реализовано как в TensorFlow, так и в PyTorch.")

    add_heading(doc, "Задания 43–45. Операция V = |sin(T) - exp(P)·√Q|", level=2)
    add_paragraph(doc, "Матрицы T, P, Q размером 3 × 10. Формула реализована "
                       "средствами Keras.ops и PyTorch. Результат — тензор "
                       "размером (3, 10).")

    add_heading(doc, "Задание 46. Разреженная матрица SciPy", level=2)
    add_paragraph(doc, "Задание пропущено: scipy.sparse.random на размере "
                       "100 000 × 100 000 уходит в бесконечную генерацию "
                       "(известная особенность реализации).")

    add_heading(doc, "Задание 47. Обратная матрица 100×100", level=2)
    add_paragraph(doc, "Случайная матрица 100×100, det = 1.16e+25. Обратная "
                       "матрица найдена, проверка M @ M_inv ≈ I: True.")

    add_heading(doc, "Задание 48. Тензор F", level=2)
    add_paragraph(doc, "Тензор F собран из исходных данных, MinMax-нормализации "
                       "и стандартизации. Форма: (198, 2, 3).")

    add_heading(doc, "Задание 49. Пайплайн sklearn: scaler + PCA", level=2)
    add_paragraph(doc, "Построен Pipeline из шагов масштабирования и PCA. "
                       "Использован критерий explained_variance_ratio_.sum(). "
                       "Оба варианта дали 1.0000 (исходных признаков два, две "
                       "компоненты объясняют всю дисперсию). Лучший вариант — minmax.")
    add_image_with_caption(doc, find_one("task49_pca_evr_*.png"),
                           "Рис. 15. Кумулятивная доля объяснённой дисперсии (log)")

    doc.add_page_break()

    # ==================== ЗАДАНИЕ 50 ====================
    add_heading(doc, "Задание 50. Обработка изображения средствами SciPy", level=1)
    add_paragraph(doc, "Взято цветное изображение размером 421 × 920 × 3 "
                       "(формат PNG). Выполнены все 10 подзаданий.")

    add_heading(doc, "1. Разделение цветовых каналов", level=2)
    add_image_with_caption(doc, find_one("task50_channels_*.png"),
                           "Рис. 16. Исходное изображение и каналы R, G, B")

    add_heading(doc, "2. Поворот на 40°", level=2)
    add_image_with_caption(doc, find_one("task50_rotated_*.png"),
                           "Рис. 17. Поворот изображения на 40° (бикубическая интерполяция)")

    add_heading(doc, "3. Преобразование в ч/б", level=2)
    add_image_with_caption(doc, find_one("task50_gray_*.png"),
                           "Рис. 18. Преобразование в оттенки серого")

    add_heading(doc, "4. Гистограмма яркостей", level=2)
    add_image_with_caption(doc, find_one("task50_gray_hist_*.png"),
                           "Рис. 19. Гистограмма ч/б изображения (256 бинов)")

    add_heading(doc, "5. Разбиение изображения на 3 части", level=2)
    add_paragraph(doc, "Изображение разбито по порогам гистограммы на тёмные, "
                       "средние и светлые области (33% и 66% накопленной частоты).")
    add_image_with_caption(doc, find_one("task50_split3_*.png"),
                           "Рис. 20. Три части: тёмные / средние / светлые области")

    add_heading(doc, "6. Чёрная круговая рамка", level=2)
    add_image_with_caption(doc, find_one("task50_framed_*.png"),
                           "Рис. 21. Изображение в чёрной круговой рамке")

    add_heading(doc, "7. Добавление случайного шума", level=2)
    add_image_with_caption(doc, find_one("task50_noisy_*.png"),
                           "Рис. 22. Добавление гауссовского шума (σ = 25)")

    add_heading(doc, "8. Гауссовское размытие", level=2)
    add_image_with_caption(doc, find_one("task50_blurred_*.png"),
                           "Рис. 23. Размытие гауссовским фильтром (σ = 2)")

    add_heading(doc, "9. Операция sharpening", level=2)
    add_image_with_caption(doc, find_one("task50_sharpened_*.png"),
                           "Рис. 24. Усиление контуров (unsharp masking, amount=1.5)")

    add_heading(doc, "10. PCA к изображению", level=2)
    add_image_with_caption(doc, find_one("task50_pca_restored_*.png"),
                           "Рис. 25. Восстановленное изображение после PCA (32 компоненты)")

    doc.add_page_break()

    # ==================== ВЫВОД ====================
    add_heading(doc, "Вывод", level=1)
    add_paragraph(doc, "В ходе выполнения лабораторной работы были изучены "
                       "основные библиотеки Python для машинного обучения и "
                       "приобретены практические навыки работы с табличными "
                       "данными.")
    add_paragraph(doc, "Освоены:")
    add_paragraph(doc, "• Чтение и запись данных в форматах .xlsx, .csv, .txt, .mat "
                       "средствами pandas и scipy.io.")
    add_paragraph(doc, "• Выявление и обработка проблем в данных (пропуски, "
                       "некорректные значения, несовпадение типов).")
    add_paragraph(doc, "• Реализация собственных функций масштабирования "
                       "(MinMax, Standard) и их сравнение с scikit-learn.")
    add_paragraph(doc, "• Статистический анализ: статистики, доверительные "
                       "интервалы, ковариация, корреляция, проверка гипотез.")
    add_paragraph(doc, "• Работа с тензорами TensorFlow, PyTorch, Keras.")
    add_paragraph(doc, "• Базовая обработка изображений средствами SciPy.")
    add_paragraph(doc, "")
    add_paragraph(doc, "Все результаты воспроизводимы благодаря фиксации seed "
                       "(set_seeds(42)). Графики и таблицы сохранены в папке "
                       "output/ с метками времени.")

    # ==================== СОХРАНЕНИЕ ====================
    doc.save(REPORT_PATH)
    print(f"Отчёт сохранён: {os.path.abspath(REPORT_PATH)}")


if __name__ == "__main__":
    main()
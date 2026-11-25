"""Функциональное распараллеливание: три разные обработки одной картинки.

Запуск: python main.py

Программа читает картинку из файла и сравнивает два способа обработки:
1. Последовательный: размытие -> удаление шума -> поиск границ.
2. Параллельный: эти три функции одновременно работают в разных процессах.

Каждая функция получает ОРИГИНАЛ и возвращает свою отдельную картинку.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from time import perf_counter

from PIL import Image, ImageOps


IMAGE_FILENAME = "example.png"

REPEATS = 3

PROJECT_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = PROJECT_FOLDER / "output"


def load_image(filename):
    """Читает фото и возвращает яркости, ширину и высоту.

    Pillow нужен только для открытия и сохранения файлов.
    convert("L") переводит цветное фото в оттенки серого.
    """
    with Image.open(filename) as original:
        image = ImageOps.exif_transpose(original).convert("L")
        width, height = image.size
        if width < 3 or height < 3:
            raise ValueError("Нужна картинка размером минимум 3 × 3 пикселя.")
        return image.tobytes(), width, height


def get_neighbors(pixels, width, x, y):
    """Возвращает 9 яркостей: текущую точку и восемь соседей."""

    neighbors = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            position = (y + dy) * width + (x + dx)
            neighbors.append(pixels[position])

    return neighbors


def blur(pixels, width, height):
    """Размытие: заменяет яркость точки средним значением девяти яркостей."""

    result = bytearray(pixels)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighbors = get_neighbors(pixels, width, x, y)
            result[y * width + x] = sum(neighbors) // 9

    return bytes(result)


def denoise(pixels, width, height):
    """Медианный фильтр: убирает одиночные чёрные и белые точки шума."""

    result = bytearray(pixels)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighbors = get_neighbors(pixels, width, x, y)
            neighbors.sort()
            result[y * width + x] = neighbors[4]

    return bytes(result)


def find_edges(pixels, width, height):
    """Выделяет границы оператором Собеля: ищет резкие перепады яркости."""

    result = bytearray(width * height)  # Сначала все точки чёрные.

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            p = get_neighbors(pixels, width, x, y)

            left = p[0] + 2 * p[3] + p[6]
            right = p[2] + 2 * p[5] + p[8]
            top = p[0] + 2 * p[1] + p[2]
            bottom = p[6] + 2 * p[7] + p[8]

            gx = right - left
            gy = bottom - top

            brightness = abs(gx) + abs(gy)
            result[y * width + x] = min(255, brightness)

    return bytes(result)


def run_sequential(pixels, width, height):
    """Выполняет три разных действия ПО ОЧЕРЕДИ над исходной картинкой."""

    blurred = blur(pixels, width, height)
    cleaned = denoise(pixels, width, height)
    edges = find_edges(pixels, width, height)

    return blurred, cleaned, edges


def run_parallel(pixels, width, height):
    """Выполняет те же действия ОДНОВРЕМЕННО в трёх отдельных процессах."""

    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        blur_task = pool.submit(blur, pixels, width, height)
        denoise_task = pool.submit(denoise, pixels, width, height)
        edges_task = pool.submit(find_edges, pixels, width, height)

        blurred = blur_task.result()
        cleaned = denoise_task.result()
        edges = edges_task.result()

    return blurred, cleaned, edges


def save_image(filename, pixels, width, height):
    """Сохраняет яркости в PNG."""

    image = Image.frombytes("L", (width, height), pixels)
    image.save(OUTPUT_FOLDER / filename)


def main():
    """Подготавливает данные, сравнивает способы запуска."""

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    image_path = PROJECT_FOLDER / IMAGE_FILENAME

    if not image_path.is_file():
        raise SystemExit(f"Картинка не найдена: {image_path}")

    pixels, width, height = load_image(image_path)
    print(f"Файл: {image_path}")
    print(f"Изображение: {width} × {height} пикселей", flush=True)
    save_image("01_original.png", pixels, width, height)

    sequential_times = []
    parallel_times = []
    report_lines = []

    for repeat in range(REPEATS):
        print(f"\nЗамер {repeat + 1} из {REPEATS}", flush=True)

        start = perf_counter()
        sequential_results = run_sequential(pixels, width, height)
        sequential_seconds = perf_counter() - start
        print(f"  Последовательно: {sequential_seconds:.3f} с", flush=True)

        start = perf_counter()
        parallel_results = run_parallel(pixels, width, height)
        parallel_seconds = perf_counter() - start
        print(f"  Параллельно:     {parallel_seconds:.3f} с", flush=True)

        if sequential_results != parallel_results:
            raise RuntimeError("Результаты двух способов запуска различаются!")

        sequential_times.append(sequential_seconds)
        parallel_times.append(parallel_seconds)
        report_lines.append(
            f"Замер {repeat + 1}: последовательно {sequential_seconds:.6f} с; "
            f"параллельно {parallel_seconds:.6f} с"
        )

    sequential_time = median(sequential_times)
    parallel_time = median(parallel_times)
    speedup = sequential_time / parallel_time

    print("\nСРАВНЕНИЕ: медианное время")
    print(f"Последовательно: {sequential_time:.3f} с")
    print(f"Параллельно:     {parallel_time:.3f} с")
    print(f"Ускорение:      {speedup:.2f} раза")
    print("Результаты всех обработок совпадают.")
    if speedup < 1:
        print("В этом запуске параллельный вариант оказался медленнее.")

    blurred, cleaned, edges = parallel_results
    save_image("02_blurred.png", blurred, width, height)
    save_image("03_denoised.png", cleaned, width, height)
    save_image("04_edges.png", edges, width, height)

    report_lines.extend([
        f"Размер: {width} x {height}",
        f"Последовательно, медиана: {sequential_time:.6f} с",
        f"Параллельно, медиана: {parallel_time:.6f} с",
        f"Ускорение: {speedup:.6f}",
        "Результаты всех обработок совпадают.",
        "Чтение картинки, запись файлов и сравнение результатов не входят в замер.",
        "Запуск процессов, передача данных и завершение процессов входят в замер.",
    ])
    report_path = OUTPUT_FOLDER / "comparison.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Картинки и отчёт: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()

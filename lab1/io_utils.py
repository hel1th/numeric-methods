from pathlib import Path

import numpy as np


def read_linear_systems(
    file_path: str | Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Считывает список СЛАУ вида (A, b) из текстового файла.

    Формат файла:
      - Блоки систем разделяются строкой '---'.
      - Внутри блока: n строк матрицы A и 1 строка вектора свободных членов b.
      - Комментарии начинаются с символа '#'.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    with path.open("r", encoding="utf-8") as f:
        raw_lines = [
            line.split("#")[0].strip() for line in f if line.split("#")[0].strip()
        ]

    blocks: list[list[list[float]]] = []
    current_block: list[list[float]] = []

    for line in raw_lines:
        if line == "---":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append([float(x) for x in line.split()])

    if current_block:
        blocks.append(current_block)

    systems: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, block in enumerate(blocks, start=1):
        if len(block) < 2:
            raise ValueError(f"Блок {idx} содержит слишком мало строк.")

        A = np.array(block[:-1], dtype=np.float64)
        b = np.array(block[-1], dtype=np.float64)

        n = len(b)
        if A.shape != (n, n):
            raise ValueError(
                f"Несоответствие размерностей в блоке {idx}: "
                f"матрица A имеет форму {A.shape}, а вектор b длины {n}."
            )

        systems.append((A, b))

    return systems


def read_tridiagonal_systems(
    file_path: str | Path,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Считывает список трехдиагональных систем (a, b, c, d) из файла.

    Формат блока (4 строки):
      Строка 1: поддиагональ a (длина n - 1)
      Строка 2: главная диагональ b (длина n)
      Строка 3: наддиагональ c (длина n - 1)
      Строка 4: правая часть d (длина n)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    with path.open("r", encoding="utf-8") as f:
        raw_lines = [
            line.split("#")[0].strip() for line in f if line.split("#")[0].strip()
        ]

    blocks: list[list[list[float]]] = []
    current_block: list[list[float]] = []

    for line in raw_lines:
        if line == "---":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append([float(x) for x in line.split()])

    if current_block:
        blocks.append(current_block)

    systems: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for idx, block in enumerate(blocks, start=1):
        if len(block) != 4:
            raise ValueError(
                f"Блок {idx} должен содержать ровно 4 строки (a, b, c, d), "
                f"но получено {len(block)}."
            )

        a = np.array(block[0], dtype=np.float64)
        b = np.array(block[1], dtype=np.float64)
        c = np.array(block[2], dtype=np.float64)
        d = np.array(block[3], dtype=np.float64)

        n = len(b)
        if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
            raise ValueError(
                f"Несоответствие размерностей в блоке {idx}: "
                f"len(b)={n}, len(a)={len(a)}, len(c)={len(c)}, len(d)={len(d)}."
            )

        systems.append((a, b, c, d))

    return systems


def read_matrices(file_path: str | Path) -> list[np.ndarray]:
    """
    Считывает список квадратных матриц из текстового файла.
    Блоки матриц разделяются строкой '---'.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    with path.open("r", encoding="utf-8") as f:
        raw_lines = [
            line.split("#")[0].strip() for line in f if line.split("#")[0].strip()
        ]

    blocks: list[list[list[float]]] = []
    current_block: list[list[float]] = []

    for line in raw_lines:
        if line == "---":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append([float(x) for x in line.split()])

    if current_block:
        blocks.append(current_block)

    matrices: list[np.ndarray] = []
    for idx, block in enumerate(blocks, start=1):
        A = np.array(block, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"Блок {idx} не является квадратной матрицей: форма {A.shape}"
            )
        matrices.append(A)

    return matrices

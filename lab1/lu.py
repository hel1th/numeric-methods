"""
Лабораторная работа 1.1: LU-разложение с выбором главного элемента по столбцу.

Реализованы функции:
  - lu_factor: вычисление LU-разложения с перестановками (PA = LU)
  - lu_det: вычисление определителя матрицы
  - lu_solve: решение СЛАУ Ax = b
  - lu_inverse: нахождение обратной матрицы A^(-1)
"""

import numpy as np


def lu_factor(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Выполняет LU-разложение квадратной матрицы A с частичным выбором
    главного элемента (по столбцу).

    Возвращает:
      L: нижнетреугольная матрица с единицами на диагонали
      U: верхнетреугольная матрица
      p: вектор перестановок строк (размер n), такой что P @ A = L @ U
      num_swaps: количество выполненных перестановок строк
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")

    n = A.shape[0]
    A_lu = A.copy()
    p = np.arange(n)
    num_swaps = 0

    for k in range(n - 1):
        # Поиск максимального по модулю элемента в k-м столбце (начиная с k-й строки)
        max_index = np.argmax(np.abs(A_lu[k:, k])) + k

        if np.isclose(A_lu[max_index, k], 0.0):
            raise ValueError(f"Матрица вырождена: нулевой ведущий элемент на шаге {k}.")

        # Перестановка строк, если максимальный элемент не на диагонали
        if max_index != k:
            A_lu[[k, max_index]] = A_lu[[max_index, k]]
            p[[k, max_index]] = p[[max_index, k]]
            num_swaps += 1

        # Исключение Гаусса (накопление L под диагональю и U на/над диагональю)
        for i in range(k + 1, n):
            A_lu[i, k] /= A_lu[k, k]
            A_lu[i, k + 1 :] -= A_lu[i, k] * A_lu[k, k + 1 :]

    # Проверяем последний диагональный элемент
    if np.isclose(A_lu[n - 1, n - 1], 0.0):
        raise ValueError("Матрица вырождена: нулевой последний диагональный элемент.")

    # Разделяем совмещенную матрицу A_lu на L и U
    L = np.tril(A_lu, -1) + np.eye(n)
    U = np.triu(A_lu)

    return L, U, p, num_swaps


def lu_det(U: np.ndarray, num_swaps: int) -> float:
    """
    Вычисляет определитель матрицы по её U-фактору и числу перестановок:
      det(A) = (-1)^num_swaps * prod(diag(U))
    """
    det = float(np.prod(np.diag(U)))
    return -det if num_swaps % 2 != 0 else det


def lu_solve(
    L: np.ndarray,
    U: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Решает СЛАУ Ax = b с использованием LU-разложения:
      1. Ly = Pb (прямой ход)
      2. Ux = y  (обратный ход)
    """
    b = np.asarray(b, dtype=np.float64)
    n = len(b)
    b_permuted = b[p]

    # Прямой ход: L y = P b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])

    # Обратный ход: U x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

    return x


def lu_inverse(L: np.ndarray, U: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Вычисляет обратную матрицу A^(-1), решая n систем A * x_k = e_k.
    """
    n = L.shape[0]
    inv_A = np.zeros((n, n), dtype=np.float64)
    eye = np.eye(n)

    for k in range(n):
        inv_A[:, k] = lu_solve(L, U, p, eye[:, k])

    return inv_A


def main() -> None:
    """Демонстрация работы алгоритма на простом примере."""
    A = np.array([[2.0, 1.0, 1.0], [4.0, 1.0, 0.0], [-2.0, 2.0, 1.0]])
    b = np.array([4.0, 5.0, 1.0])

    print("Матрица A:\n", A)
    print("Вектор b:\n", b)

    L, U, p, swaps = lu_factor(A)
    print("\nL:\n", np.round(L, 4))
    print("U:\n", np.round(U, 4))
    print("Вектор перестановок P:", p)
    print("Количество перестановок строк:", swaps)

    x = lu_solve(L, U, p, b)
    print("\nРешение системы x:\n", np.round(x, 4))
    print("Невязка Ax - b:\n", np.round(A @ x - b, 6))

    det = lu_det(U, swaps)
    print(f"\nОпределитель det(A) = {det:.4f} (np.linalg.det: {np.linalg.det(A):.4f})")

    A_inv = lu_inverse(L, U, p)
    print("\nОбратная матрица A^(-1):\n", np.round(A_inv, 4))
    print("Проверка A @ A^(-1) == I:\n", np.allclose(A @ A_inv, np.eye(len(b))))


if __name__ == "__main__":
    main()

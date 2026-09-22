"""
Лабораторная работа 1.5: QR-разложение матриц и QR-алгоритм
для нахождения собственных значений произвольных матриц.

Задание:
  - Реализовать алгоритм QR-разложения с помощью отражений Хаусхолдера.
  - На его основе реализовать QR-алгоритм поиска всех собственных значений
    произвольной вещественной матрицы (включая комплексные).
  - Входные данные: матрица A, точность eps.
"""

from typing import Any

import numpy as np


def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Выполняет QR-разложение матрицы A методом отражений Хаусхолдера:
      A = Q @ R
    где Q — ортогональная матрица (Q^T @ Q = I),
        R — верхнетреугольная матрица.
    """
    A_curr = np.asarray(A, dtype=np.float64).copy()
    m, n = A_curr.shape
    if m != n:
        raise ValueError("QR-разложение реализовано для квадратных матриц.")

    Q = np.eye(n, dtype=np.float64)
    R = A_curr.copy()

    for k in range(n - 1):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)

        if np.isclose(norm_x, 0.0):
            continue

        # Вектор нормали плоскости отражения Хаусхолдера: v = x + sign(x_0) * ||x|| * e_1
        sign = 1.0 if x[0] >= 0.0 else -1.0
        v = x.copy()
        v[0] += sign * norm_x

        norm_v = np.linalg.norm(v)
        if np.isclose(norm_v, 0.0):
            continue

        u = v / norm_v  # единичный вектор отражения: H = I - 2 * u @ u^T

        # Преобразование R: R[k:, k:] = H @ R[k:, k:] = R[k:, k:] - 2 * u * (u^T @ R[k:, k:])
        R[k:, k:] -= 2.0 * np.outer(u, u @ R[k:, k:])

        # Накопление матрицы Q: Q[:, k:] = Q[:, k:] @ H
        Q[:, k:] -= 2.0 * (Q[:, k:] @ u)[:, None] * u

    return Q, R


def _extract_eigenvalues(A: np.ndarray, eps: float = 1e-6) -> list[complex]:
    """
    Извлекает собственные значения из квазитреугольной матрицы
    (с блоками 1x1 для вещественных и 2x2 для комплексно-сопряженных пар).
    """
    n = A.shape[0]
    eigenvalues: list[complex] = []
    i = 0

    while i < n:
        # Проверяем, последний ли это элемент
        if i == n - 1:
            eigenvalues.append(complex(A[i, i], 0.0))
            i += 1
            continue

        # Проверяем поддиагональный элемент A[i+1, i]
        # Если он достаточно мал или поддиагональ ниже i+1 мала, рассматриваем блок
        subdiag_val = abs(A[i + 1, i])

        # Вычисляем дискриминант 2x2 блока
        a11, a12 = A[i, i], A[i, i + 1]
        a21, a22 = A[i + 1, i], A[i + 1, i + 1]
        trace = a11 + a22
        det = a11 * a22 - a12 * a21
        disc = (a11 - a22) ** 2 + 4.0 * a12 * a21

        # Если дискриминант отрицателен, это пара комплексных корней
        if disc < 0.0:
            re = trace / 2.0
            im = np.sqrt(-disc) / 2.0
            eigenvalues.append(complex(re, im))
            eigenvalues.append(complex(re, -im))
            i += 2
        elif subdiag_val <= eps:
            # 1x1 вещественный блок
            eigenvalues.append(complex(A[i, i], 0.0))
            i += 1
        else:
            # Вещественные корни 2x2 блока
            root1 = (trace + np.sqrt(disc)) / 2.0
            root2 = (trace - np.sqrt(disc)) / 2.0
            eigenvalues.append(complex(root1, 0.0))
            eigenvalues.append(complex(root2, 0.0))
            i += 2

    return eigenvalues


def _sort_eigenvalues(vals: list[complex]) -> np.ndarray:
    """Сортирует список комплексных чисел по вещественной, затем мнимой части."""
    return np.array(sorted(vals, key=lambda z: (round(z.real, 8), round(z.imag, 8))))


def qr_eigenvalues(
    A: np.ndarray,
    eps: float = 1e-6,
    max_iter: int = 2000,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    """
    Находит все собственные значения матрицы A с помощью QR-алгоритма.

    Параметры:
      A: квадратная вещественная матрица (n x n)
      eps: требуемая точность вычислений
      max_iter: максимальное число итераций

    Возвращает:
      eigenvalues: массив комплексных собственных значений (длина n)
      iterations: количество выполненных итераций
      history: список шагов с погрешностями
    """
    A_curr = np.asarray(A, dtype=np.float64).copy()
    n = A_curr.shape[0]

    if A_curr.ndim != 2 or n != A_curr.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")

    history: list[dict[str, Any]] = []
    prev_eigs = _sort_eigenvalues(_extract_eigenvalues(A_curr, eps=eps))

    for it in range(1, max_iter + 1):
        Q, R = qr_decomposition(A_curr)
        A_curr = R @ Q

        curr_eigs = _sort_eigenvalues(_extract_eigenvalues(A_curr, eps=eps))

        # Погрешность: максимальное изменение собственных значений между шагами
        diff = float(np.max(np.abs(curr_eigs - prev_eigs)))

        history.append(
            {
                "iteration": it,
                "eigenvalue_diff": diff,
            }
        )

        if diff <= eps and it > 5:
            return curr_eigs, it, history

        prev_eigs = curr_eigs

    return prev_eigs, max_iter, history


def main() -> None:
    """Демонстрация работы QR-разложения и QR-алгоритма."""
    print("=== 1. Демонстрация QR-разложения методом отражений Хаусхолдера ===")
    A_demo = np.array([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]])
    Q, R = qr_decomposition(A_demo)
    print("Матрица A:\n", A_demo)
    print("\nМатрица Q (ортогональная):\n", np.round(Q, 4))
    print("\nМатрица R (верхнетреугольная):\n", np.round(R, 4))
    print("\nПроверка Q @ R == A:", np.allclose(Q @ R, A_demo))
    print("Проверка Q.T @ Q == I:", np.allclose(Q.T @ Q, np.eye(3)))

    print("\n=== 2. Демонстрация QR-алгоритма для собственных значений ===")
    # Матрица с вещественными и комплексно-сопряженными собственными значениями
    A = np.array([[1.0, 3.0, 1.0], [1.0, 1.0, 4.0], [4.0, 3.0, 1.0]])
    eps = 1e-6
    print("Входная матрица A:\n", A)
    print(f"Точность eps = {eps}\n")

    eigenvalues, iterations, _ = qr_eigenvalues(A, eps=eps)
    print(f"QR-алгоритм завершен за {iterations} итераций.")
    print("Найденные собственные значения (QR-алгоритм):")
    for idx, val in enumerate(eigenvalues, start=1):
        if abs(val.imag) < 1e-8:
            print(f"  lambda_{idx} = {val.real:10.6f}")
        else:
            print(f"  lambda_{idx} = {val.real:10.6f} + {val.imag:10.6f}j")

    # Сравнение с эталоном numpy
    np_eigs = np.linalg.eigvals(A)
    np_sorted = _sort_eigenvalues(list(np_eigs))
    print("\nСобственные значения (numpy.linalg.eigvals):")
    for idx, val in enumerate(np_sorted, start=1):
        if abs(val.imag) < 1e-8:
            print(f"  lambda_{idx} = {val.real:10.6f}")
        else:
            print(f"  lambda_{idx} = {val.real:10.6f} + {val.imag:10.6f}j")


if __name__ == "__main__":
    main()

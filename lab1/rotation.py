"""
Лабораторная работа 1.4: Метод вращений Якоби для нахождения
собственных значений и собственных векторов симметрических матриц.

Задание:
  - Реализовать метод вращений в виде программы (вход: матрица A, точность eps).
  - Найти собственные значения и собственные векторы.
  - Проанализировать зависимость погрешности вычислений от числа итераций.
"""

from typing import Any

import numpy as np


def off_diagonal_norm(A: np.ndarray) -> float:
    """
    Вычисляет евклидову норму внедиагональных элементов:
      t(A) = sqrt(sum_{i < j} a_{ij}^2)
    """
    n = A.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += A[i, j] ** 2
    return float(np.sqrt(total))


def find_max_off_diagonal(A: np.ndarray) -> tuple[int, int, float]:
    """
    Находит индексы (p, q), p < q, максимального по модулю
    внедиагонального элемента симметричной матрицы A.
    """
    n = A.shape[0]
    p, q = 0, 1
    max_val = abs(A[0, 1])

    for i in range(n):
        for j in range(i + 1, n):
            val = abs(A[i, j])
            if val > max_val:
                max_val = val
                p, q = i, j

    return p, q, max_val


def jacobi_rotation(
    A: np.ndarray,
    eps: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]]]:
    """
    Решает полную проблему собственных значений для симметрической матрицы A
    методом вращений Якоби.

    Параметры:
      A: симметрическая вещественная матрица (n x n)
      eps: требуемая точность (критерий останова: t(A) <= eps)
      max_iter: предельное количество итераций

    Возвращает:
      eigenvalues: вектор собственных значений (длины n)
      eigenvectors: ортогональная матрица собственных векторов (столбцы)
      iterations: количество выполненных итераций
      history: список словарей с динамикой погрешности на каждой итерации
    """
    A_curr = np.asarray(A, dtype=np.float64).copy()
    n = A_curr.shape[0]

    if A_curr.ndim != 2 or A_curr.shape[0] != A_curr.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")

    if not np.allclose(A_curr, A_curr.T, atol=1e-8):
        raise ValueError(
            "Метод вращений Якоби применим только к симметрическим матрицам."
        )

    # Матрица собственных векторов (накапливает произведение матриц вращений)
    V = np.eye(n, dtype=np.float64)
    history: list[dict[str, Any]] = []

    for it in range(1, max_iter + 1):
        err = off_diagonal_norm(A_curr)
        p, q, max_val = find_max_off_diagonal(A_curr)

        # Вычисляем максимальную невязку собственных пар ||A v_i - lambda_i v_i||
        curr_eigvals = np.diag(A_curr)
        residuals = [
            float(np.linalg.norm(A @ V[:, i] - curr_eigvals[i] * V[:, i]))
            for i in range(n)
        ]
        max_residual = max(residuals)

        history.append(
            {
                "iteration": it,
                "off_diag_norm": err,
                "max_off_diag": max_val,
                "max_residual": max_residual,
            }
        )

        # Критерий останова: норма внедиагональных элементов <= eps
        if err <= eps:
            return np.diag(A_curr), V, it - 1, history

        # Вычисление угла вращения phi
        a_pp = A_curr[p, p]
        a_qq = A_curr[q, q]
        a_pq = A_curr[p, q]

        if np.isclose(a_pp, a_qq):
            phi = np.pi / 4.0 if a_pq > 0 else -np.pi / 4.0
        else:
            phi = 0.5 * np.arctan(2.0 * a_pq / (a_pp - a_qq))

        c = np.cos(phi)
        s = np.sin(phi)

        # Матрица плоского вращения U_k
        U = np.eye(n, dtype=np.float64)
        U[p, p] = c
        U[q, q] = c
        U[p, q] = -s
        U[q, p] = s

        # Преобразование подобия: A^(k+1) = U^T * A^(k) * U
        A_curr = U.T @ A_curr @ U

        # Накопление собственных векторов: V^(k+1) = V^(k) * U
        V = V @ U

    raise RuntimeError(f"Метод вращений не сошелся за {max_iter} итераций.")


def print_convergence_analysis(history: list[dict[str, Any]]) -> None:
    """Выводит таблицу зависимости погрешности от номера итерации."""
    print(
        f"{'Шаг':>6} | {'||Внедиаг||':>14} | {'Max |a_ij|':>14} | {'Max ||A*v - lam*v||':>22}"
    )
    print("-" * 64)
    for row in history:
        print(
            f"{row['iteration']:6d} | "
            f"{row['off_diag_norm']:14.4e} | "
            f"{row['max_off_diag']:14.4e} | "
            f"{row['max_residual']:22.4e}"
        )


def main() -> None:
    """Демонстрация работы метода вращений Якоби на тестовой симметричной матрице."""
    A = np.array([[-8.0, -1.0, 7.0], [-1.0, 7.0, 4.0], [7.0, 4.0, 2.0]])
    eps = 1e-6

    print("Входная симметрическая матрица A:\n", A)
    print(f"Точность eps = {eps}\n")

    eigenvalues, eigenvectors, iterations, history = jacobi_rotation(A, eps=eps)

    print(f"Вычислено за {iterations} итераций:")
    print("Собственные значения (метод Якоби):\n", np.round(eigenvalues, 6))

    # Сравнение с эталоном numpy
    np_vals, _ = np.linalg.eigh(A)
    print("Собственные значения (numpy.linalg.eigh):\n", np.round(np.sort(np_vals), 6))

    print("\nМатрица собственных векторов (столбцы):\n", np.round(eigenvectors, 6))

    # Проверка ортогональности V: V.T @ V == I
    ortho_check = np.allclose(eigenvectors.T @ eigenvectors, np.eye(A.shape[0]))
    print(
        f"Проверка ортогональности собственных векторов (V^T @ V == I): {ortho_check}"
    )

    # Проверка равенств A v_i = lambda_i v_i
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        residual = np.linalg.norm(A @ v - lam * v)
        print(f"  lambda_{i + 1} = {lam:10.6f}, ||A*v - lambda*v|| = {residual:.2e}")

    print("\n--- Зависимость погрешности от числа итераций ---")
    print_convergence_analysis(history)


if __name__ == "__main__":
    main()

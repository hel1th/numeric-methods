"""
Лабораторная работа 1.3: Итерационные методы решения СЛАУ.

Реализованы:
  - Метод простых итераций (метод Якоби)
  - Метод Зейделя (Гаусса-Зейделя)
"""

import numpy as np


def prepare_iteration_system(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Приводит СЛАУ Ax = b к эквивалентному итерационному виду x = alpha * x + beta.

    Формулы:
      alpha[i, j] = -A[i, j] / A[i, i]  (при i != j),  alpha[i, i] = 0
      beta[i] = b[i] / A[i, i]

    Возвращает:
      alpha: матрица итераций
      beta: вектор правых частей
      alpha_norm: бесконечная норма матрицы alpha (max по строкам суммы модулей)
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = A.shape[0]
    alpha = np.zeros((n, n), dtype=np.float64)
    beta = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if np.isclose(A[i, i], 0.0):
            raise ValueError(
                f"Нулевой диагональный элемент в строке {i}. "
                "Невозможно построить итерационный вид."
            )

        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]

    # Бесконечная норма (максимальная строчная норма)
    alpha_norm = float(np.max(np.sum(np.abs(alpha), axis=1)))
    return alpha, beta, alpha_norm


def simple_iteration(
    A: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-4,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int]:
    """
    Решает СЛАУ Ax = b методом простых итераций.

    Критерий остановки (при ||alpha|| < 1):
      ||x^(k) - x^(k-1)|| <= (1 - ||alpha||) / ||alpha|| * eps
      что эквивалентно: factor * ||x^(k) - x^(k-1)|| <= eps, где factor = ||alpha|| / (1 - ||alpha||)

    Возвращает:
      x: вектор приближенного решения
      iterations: количество выполненных итераций
    """
    alpha, beta, alpha_norm = prepare_iteration_system(A, b)
    x = beta.copy()

    # Множитель для апостериорной оценки погрешности
    factor = (alpha_norm / (1.0 - alpha_norm)) if alpha_norm < 1.0 else 1.0

    for it in range(1, max_iter + 1):
        x_next = alpha @ x + beta
        diff = float(np.max(np.abs(x_next - x)))

        if factor * diff <= eps:
            return x_next, it

        x = x_next

    raise RuntimeError(f"Метод простых итераций не сошелся за {max_iter} шагов.")


def seidel_iteration(
    A: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-4,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int]:
    """
    Решает СЛАУ Ax = b методом Зейделя.

    Каждая компонента вычисляется с использованием обновленных значений
    текущей итерации для j < i:
      x_i^(k+1) = beta_i + sum_{j < i} alpha_{ij} x_j^(k+1) + sum_{j > i} alpha_{ij} x_j^(k)

    Критерий остановки:
      factor * ||x^(k+1) - x^(k)|| <= eps,
      где factor = ||C|| / (1 - ||alpha||), C — строго верхнетреугольная часть alpha.

    Возвращает:
      x: вектор приближенного решения
      iterations: количество выполненных итераций
    """
    alpha, beta, alpha_norm = prepare_iteration_system(A, b)
    n = len(b)
    x = beta.copy()

    # Выделяем строго верхнетреугольную часть C матрицы alpha
    C = np.triu(alpha, 1)
    c_norm = float(np.max(np.sum(np.abs(C), axis=1)))

    if alpha_norm < 1.0:
        factor = (
            c_norm / (1.0 - alpha_norm)
            if c_norm > 0
            else (alpha_norm / (1.0 - alpha_norm))
        )
    else:
        factor = 1.0

    for it in range(1, max_iter + 1):
        x_next = x.copy()

        for i in range(n):
            left_sum = np.dot(alpha[i, :i], x_next[:i])
            right_sum = np.dot(alpha[i, i + 1 :], x[i + 1 :])
            x_next[i] = beta[i] + left_sum + right_sum

        diff = float(np.max(np.abs(x_next - x)))

        if factor * diff <= eps:
            return x_next, it

        x = x_next

    raise RuntimeError(f"Метод Зейделя не сошелся за {max_iter} шагов.")


def main() -> None:
    """Сравнение методов простых итераций и Зейделя."""
    A = np.array([[10.0, 1.0, -1.0], [1.0, 10.0, -1.0], [-1.0, 1.0, 10.0]])
    b = np.array([11.0, 10.0, 10.0])
    eps = 1e-5

    alpha, beta, alpha_norm = prepare_iteration_system(A, b)
    print(
        f"Норма матрицы alpha: ||alpha||_inf = {alpha_norm:.4f} (< 1: условие сходимости выполнено)"
    )

    x_simple, it_simple = simple_iteration(A, b, eps=eps)
    print(f"\nМетод простых итераций: {it_simple} шагов")
    print("Решение x:", np.round(x_simple, 5))
    print("Невязка:", np.round(A @ x_simple - b, 7))

    x_seidel, it_seidel = seidel_iteration(A, b, eps=eps)
    print(f"\nМетод Зейделя: {it_seidel} шагов")
    print("Решение x:", np.round(x_seidel, 5))
    print("Невязка:", np.round(A @ x_seidel - b, 7))


if __name__ == "__main__":
    main()

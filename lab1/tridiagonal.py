"""
Лабораторная работа 1.2: Метод прогонки для трехдиагональных СЛАУ (алгоритм Томаса).

Система уравнений вида:
  b_0 * x_0 + c_0 * x_1 = d_0
  a_{i-1} * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i,  i = 1, ..., n-2
  a_{n-2} * x_{n-2} + b_{n-1} * x_{n-1} = d_{n-1}
"""

import numpy as np


def check_diagonal_dominance(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    """
    Проверяет условие диагонального преобладания матрицы:
      |b_0| >= |c_0|
      |b_i| >= |a_{i-1}| + |c_i|,  i = 1, ..., n-2
      |b_{n-1}| >= |a_{n-2}|
    с хотя бы одним строгим неравенством (достаточное условие корректности прогонки).
    """
    n = len(b)
    has_strict = False

    # Первая строка
    if abs(b[0]) < abs(c[0]):
        return False
    if abs(b[0]) > abs(c[0]):
        has_strict = True

    # Промежуточные строки
    for i in range(1, n - 1):
        row_sum = abs(a[i - 1]) + abs(c[i])
        if abs(b[i]) < row_sum:
            return False
        if abs(b[i]) > row_sum:
            has_strict = True

    # Последняя строка
    if abs(b[n - 1]) < abs(a[n - 2]):
        return False
    if abs(b[n - 1]) > abs(a[n - 2]):
        has_strict = True

    return has_strict


def compute_residual(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Вычисляет евклидову норму вектора невязки r = Ax - d для трехдиагональной матрицы.
    """
    n = len(d)
    Ax = np.zeros(n, dtype=np.float64)

    # Первая строка
    Ax[0] = b[0] * x[0] + c[0] * x[1]

    # Промежуточные строки
    for i in range(1, n - 1):
        Ax[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]

    # Последняя строка
    Ax[n - 1] = a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1]

    residual = Ax - d
    return float(np.linalg.norm(residual))


def tridiagonal_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """
    Решает трехдиагональную СЛАУ методом прогонки.

    Формула связи: x_i = P_i * x_{i+1} + Q_i

    Параметры:
      a: поддиагональ (длина n - 1)
      b: главная диагональ (длина n)
      c: наддиагональ (длина n - 1)
      d: вектор правой части (длина n)

    Возвращает:
      x: вектор решения длины n
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n = len(d)
    if len(b) != n or len(a) != n - 1 or len(c) != n - 1:
        raise ValueError("Несоответствие размерностей входных массивов.")

    P = np.zeros(n - 1, dtype=np.float64)
    Q = np.zeros(n, dtype=np.float64)

    # --- Прямой ход: вычисление прогоночных коэффициентов ---
    if np.isclose(b[0], 0.0):
        raise ValueError("Деление на ноль: b[0] равен 0.")

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] + a[i - 1] * P[i - 1]
        if np.isclose(denom, 0.0):
            raise ValueError(f"Деление на ноль в знаменателе на шаге {i}.")

        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i - 1] * Q[i - 1]) / denom

    # Последний коэффициент Q
    denom_last = b[n - 1] + a[n - 2] * P[n - 2]
    if np.isclose(denom_last, 0.0):
        raise ValueError(f"Деление на ноль в знаменателе на последнем шаге {n - 1}.")

    Q[n - 1] = (d[n - 1] - a[n - 2] * Q[n - 2]) / denom_last

    # --- Обратный ход: нахождение вектора x ---
    x = np.zeros(n, dtype=np.float64)
    x[n - 1] = Q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def main() -> None:
    """Демонстрация метода прогонки."""
    # Пример системы:
    # 10*x0 - 2*x1            = 8
    #  1*x0 + 10*x1 - 2*x2     = 9
    #         1*x1 + 10*x2 - 2*x3 = 9
    #                1*x2 + 10*x3 = 11
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([10.0, 10.0, 10.0, 10.0])
    c = np.array([-2.0, -2.0, -2.0])
    d = np.array([8.0, 9.0, 9.0, 11.0])

    print("Проверка диагонального преобладания:", check_diagonal_dominance(a, b, c))

    x = tridiagonal_solve(a, b, c, d)
    print("Решение x:", np.round(x, 4))

    residual = compute_residual(a, b, c, d, x)
    print(f"Норма невязки ||Ax - d||: {residual:.2e}")


if __name__ == "__main__":
    main()

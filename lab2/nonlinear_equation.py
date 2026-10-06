"""
Лабораторная работа 2.1: Методы простой итерации и Ньютона.
Уравнение: ln(x + 2) - x^2 = 0 (положительный корень).
"""

import math
import matplotlib.pyplot as plt
import numpy as np


# 1. Задаем исходную функцию, её производную и итерационный вид
def f(x: float) -> float:
    """Исходное уравнение: f(x) = ln(x + 2) - x^2 = 0"""
    return math.log(x + 2.0) - x**2


def df(x: float) -> float:
    """Производная: f'(x) = 1 / (x + 2) - 2x"""
    return 1.0 / (x + 2.0) - 2.0 * x


def d2f(x: float) -> float:
    """Вторая производная: f''(x) = -1 / (x + 2)^2 - 2 (отрицательна всюду)"""
    return -1.0 / ((x + 2.0) ** 2) - 2.0


def phi(x: float) -> float:
    """
    Приведение к виду x = phi(x):
      ln(x + 2) - x^2 = 0 => x^2 = ln(x + 2) => x = sqrt(ln(x + 2))
    Берем знак '+', так как ищем положительный корень.
    """
    return math.sqrt(math.log(x + 2.0))


def dphi(x: float) -> float:
    """Производная phi'(x) = 1 / (2 * (x + 2) * sqrt(ln(x + 2)))"""
    return 1.0 / (2.0 * (x + 2.0) * math.sqrt(math.log(x + 2.0)))


# 2. Метод простой итерации
def simple_iteration(
    x0: float, eps: float = 1e-6, max_iter: int = 100
) -> tuple[float, int, list[float]]:
    """
    Решает уравнение методом простых итераций: x_{k+1} = phi(x_k).

    Возвращает:
      - root: найденный корень
      - iterations: число выполненных итераций
      - history: список значений x на каждом шаге
    """
    # Константа сжатия на [1.0, 1.1]: max |phi'(x)| достигается в x=1.0
    q = dphi(1.0)  # q approx 0.159 < 1
    factor = q / (1.0 - q)  # множитель для апостериорной оценки погрешности

    x = x0
    history = [x]

    for k in range(1, max_iter + 1):
        x_next = phi(x)
        history.append(x_next)
        diff = abs(x_next - x)

        # Критерий остановки: q / (1 - q) * |x_{k} - x_{k-1}| <= eps
        if factor * diff <= eps:
            return x_next, k, history

        x = x_next

    return x, max_iter, history


# 3. Метод Ньютона (касательных)
def newton_method(
    x0: float, eps: float = 1e-6, max_iter: int = 100
) -> tuple[float, int, list[float]]:
    """
    Решает уравнение методом Ньютона: x_{k+1} = x_k - f(x_k) / f'(x_k).

    Условие Фурье для выбора x0: f(x0) * f''(x0) > 0.
    Так как f''(x) < 0, выбираем точку, где f(x) < 0 => x0 = 1.1.

    Возвращает:
      - root: найденный корень
      - iterations: число итераций
      - history: список значений x на каждом шаге
    """
    x = x0
    history = [x]

    for k in range(1, max_iter + 1):
        f_val = f(x)
        df_val = df(x)

        if abs(df_val) < 1e-15:
            raise ZeroDivisionError("Производная обратилась в ноль!")

        x_next = x - f_val / df_val
        history.append(x_next)
        diff = abs(x_next - x)

        # Критерий остановки: |x_{k+1} - x_k| <= eps
        if diff <= eps:
            return x_next, k, history

        x = x_next

    return x, max_iter, history


# 4. Построение графиков
def plot_graphs(
    history_iter: list[float], history_newton: list[float], exact_root: float
) -> None:
    """Строит графики локализации корня и зависимости погрешности от числа шагов."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # График 1: Графическая локализация корня
    xs = np.linspace(0.0, 2.0, 300)
    ax1.plot(xs, np.log(xs + 2.0), label=r"$y = \ln(x + 2)$", color="blue")
    ax1.plot(xs, xs**2, label=r"$y = x^2$", color="orange")
    ax1.axvspan(
        1.0, 1.1, color="yellow", alpha=0.3, label="Отрезок [1.0, 1.1]"
    )
    ax1.scatter([exact_root], [exact_root**2], color="red", zorder=5)
    ax1.annotate(
        f"Корень x* ≈ {exact_root:.4f}",
        (exact_root, exact_root**2),
        xytext=(exact_root - 0.4, exact_root**2 + 0.4),
        arrowprops=dict(arrowstyle="->", color="red"),
    )
    ax1.set_title("Графическая локализация корня")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # График 2: Зависимость погрешности от номера итерации
    err_iter = [max(abs(x - exact_root), 1e-16) for x in history_iter]
    err_newton = [max(abs(x - exact_root), 1e-16) for x in history_newton]

    ax2.plot(
        range(len(err_iter)),
        np.log10(err_iter),
        marker="o",
        label="Простая итерация (линейная)",
        color="blue",
    )
    ax2.plot(
        range(len(err_newton)),
        np.log10(err_newton),
        marker="s",
        label="Метод Ньютона (квадратичная)",
        color="red",
    )
    ax2.set_title("Зависимость погрешности от числа итераций")
    ax2.set_xlabel("Номер итерации (k)")
    ax2.set_ylabel(r"$\log_{10}|x_k - x^*|$")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("lab2/lab2_1_plots.png", dpi=150)
    plt.show()


# 5. Основная программа
def main() -> None:
    print("=" * 60)
    print("Лабораторная работа 2.1: ln(x + 2) - x^2 = 0")
    print("=" * 60)

    # Локализация корня:
    # f(1.0) = ln(3) - 1 ≈ +0.0986 > 0
    # f(1.1) = ln(3.1) - 1.21 ≈ -0.0786 < 0
    # Знак меняется => положительный корень лежит в отрезке [1.0, 1.1]
    a, b = 1.0, 1.1
    print(f"1. Локализация: f({a}) = {f(a):.4f} > 0, f({b}) = {f(b):.4f} < 0")
    print("   Корень находится на отрезке [1.0, 1.1]\n")

    # Точный корень для проверки
    exact_root, _, _ = newton_method(x0=1.1, eps=1e-15)
    print(f"2. Точный корень (эталон): x* = {exact_root:.10f}\n")

    # Входная точность
    eps = 1e-6
    print(f"3. Вычисления с точностью eps = {eps}:")

    # Метод простых итераций
    x0_iter = 1.05  # берем середину отрезка
    root_iter, it_iter, hist_iter = simple_iteration(x0=x0_iter, eps=eps)
    print(f"   [Метод простых итераций] (x0 = {x0_iter}):")
    print(f"     Корень:     x = {root_iter:.8f}")
    print(f"     Итераций:   {it_iter}")
    print(f"     Невязка:    |f(x)| = {abs(f(root_iter)):.2e}")
    print(f"     Погрешность:|x - x*| = {abs(root_iter - exact_root):.2e}\n")

    # Метод Ньютона
    # По условию Фурье f(x0) * f''(x0) > 0.
    # Так как f''(x) < 0, то берем x0, где f(x) < 0 => x0 = 1.1
    x0_newton = 1.1
    root_newton, it_newton, hist_newton = newton_method(x0=x0_newton, eps=eps)
    print(f"   [Метод Ньютона] (x0 = {x0_newton}):")
    print(f"     Корень:     x = {root_newton:.8f}")
    print(f"     Итераций:   {it_newton}")
    print(f"     Невязка:    |f(x)| = {abs(f(root_newton)):.2e}")
    print(
        f"     Погрешность:|x - x*| = {abs(root_newton - exact_root):.2e}\n"
    )

    # 4. Анализ зависимости погрешности от количества итераций
    print("=" * 60)
    print("Таблица сходимости (погрешность |x_k - x*| на каждом шаге):")
    print(
        f"{'Шаг k':<6} | {'x_k (Итерации)':<15} | {'Погрешность':<12} | {'x_k (Ньютон)':<15} | {'Погрешность':<12}"
    )
    print("-" * 60)

    max_steps = max(len(hist_iter), len(hist_newton))
    for k in range(max_steps):
        # Простая итерация
        if k < len(hist_iter):
            xi = hist_iter[k]
            ei = f"{abs(xi - exact_root):.2e}"
            xi_str = f"{xi:.8f}"
        else:
            xi_str, ei = "-", "-"

        # Ньютон
        if k < len(hist_newton):
            xn = hist_newton[k]
            en = f"{abs(xn - exact_root):.2e}"
            xn_str = f"{xn:.8f}"
        else:
            xn_str, en = "-", "-"

        print(f"{k:<6} | {xi_str:<15} | {ei:<12} | {xn_str:<15} | {en:<12}")
    print("=" * 60)

    # 5. Сравнение числа итераций для разных eps
    print("\nСравнение числа итераций в зависимости от eps:")
    print(
        f"{'eps':<10} | {'Итераций (Простая)':<20} | {'Итераций (Ньютон)':<20}"
    )
    print("-" * 55)
    for test_eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        _, k_iter, _ = simple_iteration(x0=x0_iter, eps=test_eps)
        _, k_newton, _ = newton_method(x0=x0_newton, eps=test_eps)
        print(f"{test_eps:<10.0e} | {k_iter:<20} | {k_newton:<20}")
    print("=" * 60)

    # Построение графиков
    try:
        plot_graphs(hist_iter, hist_newton, exact_root)
        print("\nГрафик сохранен в файл lab2/lab2_1_plots.png")
    except Exception as e:
        print(f"Не удалось открыть окно графика: {e}")


if __name__ == "__main__":
    main()

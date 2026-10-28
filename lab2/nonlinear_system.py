
import math
import sys
import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

A = 3.0


def f1(x1: float, x2: float) -> float:
    return (x1**2 + A**2) * x2 - A**3


def f2(x1: float, x2: float) -> float:
    return (x1 - A / 2.0) ** 2 + (x2 - A / 2.0) ** 2 - A**2


def jacobian(x1: float, x2: float) -> np.ndarray:
    j11 = 2.0 * x1 * x2
    j12 = x1**2 + A**2
    j21 = 2.0 * (x1 - A / 2.0)
    j22 = 2.0 * (x2 - A / 2.0)
    return np.array([[j11, j12], [j21, j22]], dtype=np.float64)


def phi1(x1: float, x2: float) -> float:
    val = A**2 - (x2 - A / 2.0) ** 2
    if val < 0:
        val = 0.0
    return A / 2.0 + math.sqrt(val)


def phi2(x1: float, x2: float) -> float:
    return (A**3) / (x1**2 + A**2)


# Метод простой итерации для системы
def simple_iteration_system(
    x0: tuple[float, float], eps: float = 1e-6, max_iter: int = 100
) -> tuple[np.ndarray, int, list[np.ndarray]]:
    q = 0.35
    factor = q / (1.0 - q)

    x = np.array(x0, dtype=np.float64)
    history = [x.copy()]

    for k in range(1, max_iter + 1):
        x_next = np.array([phi1(x[0], x[1]), phi2(x[0], x[1])])
        history.append(x_next.copy())

        diff = float(np.max(np.abs(x_next - x)))

        if factor * diff <= eps:
            return x_next, k, history

        x = x_next

    return x, max_iter, history


def newton_method_system(
    x0: tuple[float, float], eps: float = 1e-6, max_iter: int = 100
) -> tuple[np.ndarray, int, list[np.ndarray]]:
    x = np.array(x0, dtype=np.float64)
    history = [x.copy()]

    for k in range(1, max_iter + 1):
        F_val = np.array([f1(x[0], x[1]), f2(x[0], x[1])], dtype=np.float64)
        J = jacobian(x[0], x[1])
        
        try:
            delta_x = np.linalg.solve(J, -F_val)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"Матрица Якоби вырождена на шаге {k}!")

        x_next = x + delta_x
        history.append(x_next.copy())

        diff = float(np.max(np.abs(delta_x)))

        if diff <= eps:
            return x_next, k, history

        x = x_next

    return x, max_iter, history


# Построение графиков
def plot_system(
    history_iter: list[np.ndarray],
    history_newton: list[np.ndarray],
    exact_root: np.ndarray,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    xs = np.linspace(-1.0, 5.0, 400)

    ys_curve = 27.0 / (xs**2 + 9.0)
    ax1.plot(
        xs,
        ys_curve,
        label=r"$(x_1^2 + 9)x_2 = 27$ (кривая 1)",
        color="blue",
        linewidth=2,
    )

    angles = np.linspace(0, 2 * np.pi, 300)
    circle_x = 1.5 + 3.0 * np.cos(angles)
    circle_y = 1.5 + 3.0 * np.sin(angles)
    ax1.plot(
        circle_x,
        circle_y,
        label=r"$(x_1 - 1.5)^2 + (x_2 - 1.5)^2 = 9$ (окружность)",
        color="orange",
        linewidth=2,
    )

    ax1.scatter(
        [exact_root[0]],
        [exact_root[1]],
        color="red",
        s=80,
        zorder=6,
        label=f"Корень ({exact_root[0]:.4f}, {exact_root[1]:.4f})",
    )
    
    newt_pts = np.array(history_newton)
    ax1.plot(
        newt_pts[:, 0],
        newt_pts[:, 1],
        "r--o",
        markersize=5,
        alpha=0.8,
        label="Шаги Ньютона",
    )

    ax1.set_xlim(-0.5, 5.0)
    ax1.set_ylim(-0.5, 5.0)
    ax1.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax1.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax1.set_title("Графическая локализация решения системы", fontsize=12)
    ax1.set_xlabel("$x_1$", fontsize=11)
    ax1.set_ylabel("$x_2$", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    
    err_iter = [
        max(float(np.max(np.abs(pt - exact_root))), 1e-16)
        for pt in history_iter
    ]
    err_newton = [
        max(float(np.max(np.abs(pt - exact_root))), 1e-16)
        for pt in history_newton
    ]

    ax2.plot(
        range(len(err_iter)),
        np.log10(err_iter),
        marker="o",
        label="Простая итерация (линейная)",
        color="blue",
        linewidth=2,
    )
    ax2.plot(
        range(len(err_newton)),
        np.log10(err_newton),
        marker="s",
        label="Метод Ньютона (квадратичная)",
        color="red",
        linewidth=2,
    )

    ax2.set_title("Зависимость погрешности от числа итераций", fontsize=12)
    ax2.set_xlabel("Номер итерации (k)", fontsize=11)
    ax2.set_ylabel(r"$\log_{10}||x_k - x^*||_\infty$", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("lab2/lab2_2_plots.png", dpi=150)
    plt.show()


def main() -> None:
    print("=" * 70)
    print("Лабораторная работа 2.2: Решение системы нелинейных уравнений")
    print("  1) (x_1^2 + 9) * x_2 - 27 = 0")
    print("  2) (x_1 - 1.5)^2 + (x_2 - 1.5)^2 - 9 = 0")
    print("=" * 70)

    # Окружность с центром (1.5, 1.5) и R = 3.
    # Кривая x2 = 27 / (x1^2 + 9) пересекает окружность в 1-й четверти при x1 = 4.4, x2 = 0.9.
    x0 = (4.0, 1.0)
    print(f"\nНачальное приближение по графику: x0 = ({x0[0]}, {x0[1]})")

    exact_root, _, _ = newton_method_system(x0=x0, eps=1e-14)
    print(
        f"Точное решение: x1* = {exact_root[0]:.10f}, x2* = {exact_root[1]:.10f}\n"
    )

    eps = 1e-6
    print(f"Вычисления с точностью eps = {eps}:")

    # Метод простых итераций
    root_iter, it_iter, hist_iter = simple_iteration_system(x0=x0, eps=eps)
    res1_i = abs(f1(root_iter[0], root_iter[1]))
    res2_i = abs(f2(root_iter[0], root_iter[1]))
    err_i = float(np.max(np.abs(root_iter - exact_root)))
    print(f"   [Метод простых итераций] (шагов: {it_iter}):")
    print(f"     x1 = {root_iter[0]:.8f},  x2 = {root_iter[1]:.8f}")
    print(f"     Невязка: |f1| = {res1_i:.2e}, |f2| = {res2_i:.2e}")
    print(f"     Погрешность ||x - x*|| = {err_i:.2e}\n")

    # Метод Ньютона
    root_newt, it_newt, hist_newt = newton_method_system(x0=x0, eps=eps)
    res1_n = abs(f1(root_newt[0], root_newt[1]))
    res2_n = abs(f2(root_newt[0], root_newt[1]))
    err_n = float(np.max(np.abs(root_newt - exact_root)))
    print(f"   [Метод Ньютона] (шагов: {it_newt}):")
    print(f"     x1 = {root_newt[0]:.8f},  x2 = {root_newt[1]:.8f}")
    print(f"     Невязка: |f1| = {res1_n:.2e}, |f2| = {res2_n:.2e}")
    print(f"     Погрешность ||x - x*|| = {err_n:.2e}\n")

    
    print("=" * 70)
    print("Таблица сходимости (погрешность ||x_k - x*|| на каждом шаге):")
    print(
        f"{'Шаг k':<6} | {'(x1, x2) Простая итерация':<28} | {'Погрешность':<12} | {'Погрешность Ньютона':<12}"
    )
    print("-" * 70)

    max_steps = max(len(hist_iter), len(hist_newt))
    for k in range(max_steps):
        # Итерации
        if k < len(hist_iter):
            xi = hist_iter[k]
            ei = f"{float(np.max(np.abs(xi - exact_root))):.2e}"
            xi_str = f"({xi[0]:.4f}, {xi[1]:.4f})"
        else:
            xi_str, ei = "-", "-"

        # Ньютон
        if k < len(hist_newt):
            xn = hist_newt[k]
            en = f"{float(np.max(np.abs(xn - exact_root))):.2e}"
        else:
            en = "-"

        print(f"{k:<6} | {xi_str:<28} | {ei:<12} | {en:<12}")
    print("=" * 70)

    print("\nСравнение числа итераций в зависимости от eps:")
    print(
        f"{'eps':<10} | {'Итераций (Простая)':<20} | {'Итераций (Ньютон)':<20}"
    )
    print("-" * 55)
    for test_eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        _, k_iter, _ = simple_iteration_system(x0=x0, eps=test_eps)
        _, k_newton, _ = newton_method_system(x0=x0, eps=test_eps)
        print(f"{test_eps:<10.0e} | {k_iter:<20} | {k_newton:<20}")
    print("=" * 70)

    try:
        plot_system(hist_iter, hist_newt, exact_root)
        print("\nГрафик сохранен в файл lab2/lab2_2_plots.png")
    except Exception as e:
        print(f"Не удалось открыть окно графика: {e}")


if __name__ == "__main__":
    main()

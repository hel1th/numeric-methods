from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from lab1.io_utils import read_linear_systems
except ImportError:
    from io_utils import read_linear_systems


def prepare_iteration_system(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
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

    alpha_norm = float(np.max(np.sum(np.abs(alpha), axis=1)))
    return alpha, beta, alpha_norm


def simple_iteration(
    A: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-4,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int]:

    alpha, beta, alpha_norm = prepare_iteration_system(A, b)
    x = beta.copy()

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
    alpha, beta, alpha_norm = prepare_iteration_system(A, b)
    n = len(b)
    x = beta.copy()

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
            right_sum = np.dot(alpha[i, i + 1:], x[i + 1:])
            x_next[i] = beta[i] + left_sum + right_sum

        diff = float(np.max(np.abs(x_next - x)))

        if factor * diff <= eps:
            return x_next, it

        x = x_next

    raise RuntimeError(f"Метод Зейделя не сошелся за {max_iter} шагов.")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "iterations.txt"
    if not data_file.exists():
        raise FileNotFoundError(f"Файл с входными данными не найден: {data_file}")

    eps = 1e-6

    print("=" * 70)
    print("Лабораторная работа 1.3: Метод простых итераций (Якоби) и метод Зейделя")
    print(f"Чтение входных данных из файла: {data_file}")
    print("=" * 70)

    systems = read_linear_systems(data_file)
    for idx, (A, b) in enumerate(systems, start=1):
        n = A.shape[0]
        print(f"\n>>> СИСТЕМА {idx} (размерность n = {n}) <<<")
        print("Матрица A:\n", A)
        print("Вектор b:\n", b)

        alpha, beta, alpha_norm = prepare_iteration_system(A, b)
        print(f"Норма матрицы alpha: ||alpha||_inf = {alpha_norm:.4f}")
        if alpha_norm < 1.0:
            print("  Условие сходимости ||alpha|| < 1 выполнено")
        else:
            print("  Внимание: ||alpha|| >= 1, достаточный критерий сходимости не выполнен.")

        print(f"\nЗаданная точность вычислений: eps = {eps:.1e}")

        x_simple, it_simple = simple_iteration(A, b, eps=eps)
        x_seidel, it_seidel = seidel_iteration(A, b, eps=eps)

        print("\nКоличество выполненных итераций:")
        print(f"   Метод простых итераций (Якоби): {it_simple} шагов")
        print(f"   Метод Зейделя:                  {it_seidel} шагов")

        x_exact = np.linalg.solve(A, b)
        print("\nРешения системы:")
        print("   Метод простых итераций: x =", np.round(x_simple, 6))
        print("     Невязка ||Ax - b||:", np.max(np.abs(A @ x_simple - b)))
        print("   Метод Зейделя:          x =", np.round(x_seidel, 6))
        print("     Невязка ||Ax - b||:", np.max(np.abs(A @ x_seidel - b)))
        print("   Эталонное решение (NumPy): x =", np.round(x_exact, 6))
        print(f"     Отклонение простой итерации от эталона: {np.max(np.abs(x_simple - x_exact)):.2e}")
        print(f"     Отклонение метода Зейделя от эталона:   {np.max(np.abs(x_seidel - x_exact)):.2e}")

        print("\nСравнение методов по скорости сходимости:")
        speedup = it_simple / it_seidel if it_seidel > 0 else 1.0
        print(f"   Отношение числа шагов (Якоби / Зейдель): {speedup:.2f}x")
        if it_seidel < it_simple:
            print(f"   -> Метод Зейделя сошелся быстрее метода Якоби на {it_simple - it_seidel} итераций.")
        elif it_seidel == it_simple:
            print("   -> Оба метода потребовали одинаковое число итераций.")
        else:
            print("   -> Метод Якоби сошелся быстрее метода Зейделя.")
        print("=" * 70)


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
from typing import Any
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from lab1.io_utils import read_matrices
except ImportError:
    from io_utils import read_matrices


def off_diagonal_norm(A: np.ndarray) -> float:
    n = A.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += A[i, j] ** 2
    return float(np.sqrt(total))


def find_max_off_diagonal(A: np.ndarray) -> tuple[int, int, float]:
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
    A_curr = np.asarray(A, dtype=np.float64).copy()
    n = A_curr.shape[0]

    if A_curr.ndim != 2 or A_curr.shape[0] != A_curr.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")

    if not np.allclose(A_curr, A_curr.T, atol=1e-8):
        raise ValueError(
            "Метод вращений Якоби применим только к симметрическим матрицам."
        )

    V = np.eye(n, dtype=np.float64)
    history: list[dict[str, Any]] = []

    for it in range(1, max_iter + 1):
        err = off_diagonal_norm(A_curr)
        p, q, max_val = find_max_off_diagonal(A_curr)

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

        if err <= eps:
            return np.diag(A_curr), V, it - 1, history

        a_pp = A_curr[p, p]
        a_qq = A_curr[q, q]
        a_pq = A_curr[p, q]

        if np.isclose(a_pp, a_qq):
            phi = np.pi / 4.0 if a_pq > 0 else -np.pi / 4.0
        else:
            phi = 0.5 * np.arctan(2.0 * a_pq / (a_pp - a_qq))

        c = np.cos(phi)
        s = np.sin(phi)

        U = np.eye(n, dtype=np.float64)
        U[p, p] = c
        U[q, q] = c
        U[p, q] = -s
        U[q, p] = s

        A_curr = U.T @ A_curr @ U

        V = V @ U

    raise RuntimeError(f"Метод вращений не сошелся за {max_iter} итераций.")


def print_convergence_analysis(history: list[dict[str, Any]]) -> None:
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
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "rotation.txt"
    if not data_file.exists():
        raise FileNotFoundError(f"Файл с входными данными не найден: {data_file}")

    eps = 1e-6

    print("=" * 70)
    print("Лабораторная работа 1.4: Метод вращений Якоби")
    print(f"Чтение входных данных из файла: {data_file}")
    print("=" * 70)

    matrices = read_matrices(data_file)
    for idx, A in enumerate(matrices, start=1):
        n = A.shape[0]
        print(f"\n>>> МАТРИЦА {idx} (размерность n = {n}) <<<")
        print("Входная симметрическая матрица A:\n", A)

        if not np.allclose(A, A.T):
            print("Предупреждение: матрица не симметрична, метод Якоби требует симметричности!")

        print(f"\nЗаданная точность вычислений: eps = {eps:.1e}")

        eigenvalues, V, iterations, history = jacobi_rotation(A, eps=eps)

        print(f"\nНайденные собственные значения (итераций: {iterations}):")
        for i, val in enumerate(eigenvalues, start=1):
            print(f"   lambda_{i} = {val:12.6f}")

        np_vals, _ = np.linalg.eigh(A)
        print("   Эталонные значения (NumPy eigh):", np.round(np.sort(np_vals), 6))

        print("\nМатрица собственных векторов V (столбцы — ортонормированные векторы):")
        print(np.round(V, 6))
        for i in range(n):
            print(f"   Собственный вектор v_{i+1} для lambda_{i+1}: {np.round(V[:, i], 6)}")

        
        Lambda = np.diag(eigenvalues)
        AV = A @ V
        VLambda = V @ Lambda
        diff_matrix = np.max(np.abs(AV - VLambda))

        print("\nПроверка матричного равенства A · V = V · Lambda:")
        print("   Матрица A · V:\n", np.round(AV, 6))
        print("   Матрица V · Lambda:\n", np.round(VLambda, 6))
        print(f"   Максимальное отклонение ||A*V - V*Lambda||: {diff_matrix:.2e}")

        ortho_diff = np.max(np.abs(V.T @ V - np.eye(n)))
        print(f"   Проверка ортогональности базиса ||V^T * V - I||: {ortho_diff:.2e}")

        print("\n--- Зависимость погрешности от числа итераций ---")
        print_convergence_analysis(history)
        print("=" * 70)


if __name__ == "__main__":
    main()

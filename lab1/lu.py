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


def lu_factor(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")

    n = A.shape[0]
    A_lu = A.copy()
    p = np.arange(n)
    num_swaps = 0

    for k in range(n - 1):
        max_index = np.argmax(np.abs(A_lu[k:, k])) + k

        if np.isclose(A_lu[max_index, k], 0.0):
            raise ValueError(
                f"Матрица вырождена: нулевой ведущий элемент на шаге {k}.")

        if max_index != k:
            A_lu[[k, max_index]] = A_lu[[max_index, k]]
            p[[k, max_index]] = p[[max_index, k]]
            num_swaps += 1

        for i in range(k + 1, n):
            A_lu[i, k] /= A_lu[k, k]
            A_lu[i, k + 1:] -= A_lu[i, k] * A_lu[k, k + 1:]

    if np.isclose(A_lu[n - 1, n - 1], 0.0):
        raise ValueError(
            "Матрица вырождена: нулевой последний диагональный элемент.")

    L = np.tril(A_lu, -1) + np.eye(n)
    U = np.triu(A_lu)

    return L, U, p, num_swaps


def lu_det(U: np.ndarray, num_swaps: int) -> float:
    det = float(np.prod(np.diag(U)))
    return -det if num_swaps % 2 != 0 else det


def lu_solve(
    L: np.ndarray,
    U: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    n = len(b)
    b_permuted = b[p]

    # Прямой ход L y = P b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])

    # Обратный ход U x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def lu_inverse(L: np.ndarray, U: np.ndarray, p: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    inv_A = np.zeros((n, n), dtype=np.float64)
    eye = np.eye(n)

    for k in range(n):
        inv_A[:, k] = lu_solve(L, U, p, eye[:, k])

    return inv_A


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        __file__).parent / "data" / "lu.txt"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Файл с входными данными не найден: {data_file}")

    print("=" * 70)
    print("Лабораторная работа 1.1: LU-разложение с выбором опорного элемента")
    print(f"Чтение входных данных из файла: {data_file}")
    print("=" * 70)

    systems = read_linear_systems(data_file)
    for idx, (A, b) in enumerate(systems, start=1):
        n = A.shape[0]
        print(f"\n>>> СИСТЕМА {idx} (размерность n = {n}) <<<")
        print("Исходная матрица A:")
        print(A)
        print("Вектор правых частей b:")
        print(b)

        L, U, p, swaps = lu_factor(A)

        print("\nМатрица L (нижнетреугольная с 1 на диагонали):")
        print(np.round(L, 6))
        print("   Матрица U (верхнетреугольная):")
        print(np.round(U, 6))

        LU_prod = L @ U
        print("\nРезультат умножения L · U:")
        print(np.round(LU_prod, 6))

        x = lu_solve(L, U, p, b)
        print("\nРешение системы Ax = b:")
        print("   x =", np.round(x, 6))
        print("   Невязка ||Ax - b||:", np.max(np.abs(A @ x - b)))

        A_inv = lu_inverse(L, U, p)
        print("\nОбратная матрица A^(-1):")
        print(np.round(A_inv, 6))

        det = lu_det(U, swaps)
        print(
            f"\nОпределитель det(A) = {det:.6f} (np.linalg.det: {np.linalg.det(A):.6f})")

        AA_inv = A @ A_inv
        diff_inv = np.max(np.abs(AA_inv - np.eye(n)))
        print("\nПроверка A · A^(-1) = E:")
        print(np.round(AA_inv, 6))
        print(f"   Максимальное отклонение ||A * A^(-1) - E||: {diff_inv:.2e}")

        P = np.eye(n)[p]
        PA = P @ A
        diff_pa = np.max(np.abs(LU_prod - PA))
        print("\nПроверка L · U = P · A:")
        print("   Матрица P · A (строки A с учетом перестановок):")
        print(np.round(PA, 6))
        print(f"   Максимальное отклонение ||L*U - P*A||: {diff_pa:.2e}")
        print("=" * 70)


if __name__ == "__main__":
    main()


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

# Гарантируем корректный вывод на консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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

        sign = 1.0 if x[0] >= 0.0 else -1.0
        v = x.copy()
        v[0] += sign * norm_x

        norm_v = np.linalg.norm(v)
        if np.isclose(norm_v, 0.0):
            continue

        u = v / norm_v

        R[k:, k:] -= 2.0 * np.outer(u, u @ R[k:, k:])

        Q[:, k:] -= 2.0 * (Q[:, k:] @ u)[:, None] * u

    return Q, R


def _extract_eigenvalues(A: np.ndarray, eps: float = 1e-6) -> list[complex]:
    n = A.shape[0]
    eigenvalues: list[complex] = []
    i = 0

    while i < n:
        if i == n - 1:
            eigenvalues.append(complex(A[i, i], 0.0))
            i += 1
            continue

        subdiag_val = abs(A[i + 1, i])

        a11, a12 = A[i, i], A[i, i + 1]
        a21, a22 = A[i + 1, i], A[i + 1, i + 1]
        trace = a11 + a22
        det = a11 * a22 - a12 * a21
        disc = (a11 - a22) ** 2 + 4.0 * a12 * a21

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
    """Демонстрация работы согласно требованиям задания 1.5."""
    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        __file__).parent / "data" / "qr.txt"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Файл с входными данными не найден: {data_file}")

    eps = 1e-6

    print("=" * 70)
    print("Лабораторная работа 1.5: QR-разложение и собственные значения")
    print(f"Чтение входных данных из файла: {data_file}")
    print("=" * 70)

    matrices = read_matrices(data_file)
    for idx, A in enumerate(matrices, start=1):
        n = A.shape[0]
        print(f"\n>>> МАТРИЦА {idx} (размерность n = {n}) <<<")

        print("\nИсходная матрица A:")
        print(A)

        Q, R = qr_decomposition(A)
        print("\nМатрица Q (ортогональная, Q^T * Q = I):")
        print(np.round(Q, 6))
        print("\n   Матрица R (верхнетреугольная):")
        print(np.round(R, 6))

        QR_product = Q @ R
        print("\nРезультат умножения Q · R:")
        print(np.round(QR_product, 6))
        diff_qr = np.max(np.abs(QR_product - A))
        print(f"   Максимальное отклонение ||Q*R - A||: {diff_qr:.2e}")

        eigenvalues, iterations, _ = qr_eigenvalues(A, eps=eps)
        has_complex = any(abs(val.imag) > 1e-6 for val in eigenvalues)
        print(
            f"\nНайденные собственные значения (QR-алгоритм, итераций: {iterations}):")
        if has_complex:
            print("   -> Обнаружена комплексно-сопряженная пара собственных значений!")
        for i, val in enumerate(eigenvalues, start=1):
            if abs(val.imag) < 1e-8:
                print(f"   lambda_{i} = {val.real:10.6f}")
            else:
                sign = "+" if val.imag >= 0 else "-"
                print(
                    f"   lambda_{i} = {val.real:10.6f} {sign} {abs(val.imag):.6f}i  (комплексная пара)")

        print("\nПроверка через NumPy (np.linalg.eigvals):")
        np_eigs = _sort_eigenvalues(list(np.linalg.eigvals(A)))
        for i, val in enumerate(np_eigs, start=1):
            if abs(val.imag) < 1e-8:
                print(f"   lambda_{i} = {val.real:10.6f}")
            else:
                sign = "+" if val.imag >= 0 else "-"
                print(
                    f"   lambda_{i} = {val.real:10.6f} {sign} {abs(val.imag):.6f}i")
        print("=" * 70)


if __name__ == "__main__":
    main()

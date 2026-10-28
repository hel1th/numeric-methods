
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


def _extract_eigenvalues(
    A: np.ndarray, eps: float
) -> tuple[list[complex], bool]:
    """Извлечение собственных значений из квазитреугольной матрицы.

    Возвращает список собственных значений и флаг развязки блоков.
    """
    n = A.shape[0]
    eigenvalues: list[complex] = []
    converged = True
    i = 0

    while i < n:
        if i == n - 1:
            eigenvalues.append(complex(A[i, i]))
            i += 1
        elif abs(A[i + 1, i]) <= eps:
            if i + 2 < n and np.max(np.abs(A[i + 2 :, i])) > eps:
                converged = False
            eigenvalues.append(complex(A[i, i]))
            i += 1
        else:
            if i + 2 < n:
                if (
                    np.max(np.abs(A[i + 2 :, i])) > eps
                    or np.max(np.abs(A[i + 2 :, i + 1])) > eps
                ):
                    converged = False

            a11, a12 = A[i, i], A[i, i + 1]
            a21, a22 = A[i + 1, i], A[i + 1, i + 1]
            tr = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = tr**2 - 4.0 * det

            if disc < 0:
                re = tr / 2.0
                im = np.sqrt(-disc) / 2.0
                eigenvalues.append(complex(re, im))
                eigenvalues.append(complex(re, -im))
            else:
                converged = False
                eigenvalues.append(complex((tr + np.sqrt(disc)) / 2.0))
                eigenvalues.append(complex((tr - np.sqrt(disc)) / 2.0))
            i += 2

    eigenvalues.sort(key=lambda x: (round(x.real, 8), round(x.imag, 8)))
    return eigenvalues, converged


def qr_eigenvalues(
    A: np.ndarray,
    eps: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    A_curr = np.asarray(A, dtype=np.float64).copy()
    n = A_curr.shape[0]

    if A_curr.ndim != 2 or n != A_curr.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")

    history: list[dict[str, Any]] = []
    prev_eigs: list[complex] | None = None

    for it in range(1, max_iter + 1):
        Q, R = qr_decomposition(A_curr)
        A_curr = R @ Q

        curr_eigs, is_converged = _extract_eigenvalues(A_curr, eps)

        diff = None
        if prev_eigs is not None and is_converged:
            diff = max(abs(c - p) for c, p in zip(curr_eigs, prev_eigs))
            history.append({"iteration": it, "diff": diff})
            if diff <= eps:
                res = np.array(curr_eigs)
                if np.all(np.abs(res.imag) <= eps):
                    res = res.real
                return res, it, history
        else:
            history.append({"iteration": it, "diff": None})

        prev_eigs = curr_eigs

    res = np.array(_extract_eigenvalues(A_curr, eps)[0])
    if np.all(np.abs(res.imag) <= eps):
        res = res.real
    return res, max_iter, history


def _format_eigenvalue(val: complex | float, eps: float = 1e-6) -> str:
    if isinstance(val, (complex, np.complexfloating)) and abs(val.imag) > eps:
        sign = "+" if val.imag >= 0 else "-"
        return f"{val.real:10.6f} {sign} {abs(val.imag):.6f}i"
    real_val = val.real if isinstance(val, (complex, np.complexfloating)) else float(val)
    return f"{real_val:10.6f}"


def main() -> None:
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
        print(
            f"\nНайденные собственные значения (QR-алгоритм, итераций: {iterations}):")
        for i, val in enumerate(eigenvalues, start=1):
            print(f"   lambda_{i} = {_format_eigenvalue(val, eps)}")

        print("\nПроверка через NumPy (np.linalg.eigvals):")
        np_eigs = sorted(
            np.linalg.eigvals(A),
            key=lambda x: (round(x.real, 8), round(x.imag, 8)),
        )
        for i, val in enumerate(np_eigs, start=1):
            print(f"   lambda_{i} = {_format_eigenvalue(val, eps)}")
        print("=" * 70)


if __name__ == "__main__":
    main()

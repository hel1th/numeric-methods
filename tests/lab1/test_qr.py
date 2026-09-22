"""
Тестирование лабораторной работы 1.5 (QR-разложение и QR-алгоритм).
"""

from pathlib import Path

import numpy as np

from lab1.io_utils import read_matrices
from lab1.qr import _sort_eigenvalues, qr_decomposition, qr_eigenvalues


def test_qr_decomposition() -> None:
    test_file = Path(__file__).parent / "test_qr.txt"
    matrices = read_matrices(test_file)

    print(f"\nТестирование алгоритма QR-разложения методом отражений Хаусхолдера...")

    for idx, A in enumerate(matrices, start=1):
        n = A.shape[0]
        Q, R = qr_decomposition(A)

        # 1. Проверка ортогональности Q: Q^T @ Q == I
        assert np.allclose(Q.T @ Q, np.eye(n), atol=1e-8), (
            f"Матрица Q не ортогональна на тесте {idx}"
        )

        # 2. Проверка восстановления матрицы A: Q @ R == A
        assert np.allclose(Q @ R, A, atol=1e-8), f"Q @ R != A на тесте {idx}"

        # 3. Проверка верхнетреугольного вида R
        assert np.allclose(R, np.triu(R), atol=1e-8), (
            f"Матрица R не является верхнетреугольной на тесте {idx}"
        )

        print(
            f"✓ Тест {idx}: QR-разложение корректно (Q ортогональна, R треугольна, Q*R == A)"
        )


def test_qr_eigenvalues() -> None:
    test_file = Path(__file__).parent / "test_qr.txt"
    matrices = read_matrices(test_file)
    eps = 1e-5

    print(
        f"\nТестирование QR-алгоритма вычисления собственных значений (eps = {eps})..."
    )

    for idx, A in enumerate(matrices, start=1):
        our_eigs, iters, _ = qr_eigenvalues(A, eps=eps)

        np_eigs = _sort_eigenvalues(list(np.linalg.eigvals(A)))

        # Сравнение собственных значений с эталоном numpy
        assert np.allclose(our_eigs, np_eigs, atol=1e-3), (
            f"Несовпадение собственных значений на тесте {idx}:\n"
            f"QR:    {our_eigs}\nNumpy: {np_eigs}"
        )

        print(
            f"✓ Тест {idx}: собственные значения найдены за {iters} итераций и совпадают с numpy"
        )


if __name__ == "__main__":
    test_qr_decomposition()
    test_qr_eigenvalues()

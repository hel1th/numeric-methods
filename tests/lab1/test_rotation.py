"""
Тестирование лабораторной работы 1.4 (Метод вращений Якоби).
"""

from pathlib import Path

import numpy as np

from lab1.io_utils import read_matrices
from lab1.rotation import jacobi_rotation, off_diagonal_norm


def test_jacobi_rotation() -> None:
    test_file = Path(__file__).parent / "test_rotation.txt"
    matrices = read_matrices(test_file)
    eps = 1e-6

    print(
        f"\nЗапуск тестов метода вращений Якоби из {test_file.name} (eps = {eps})...\n"
    )

    for idx, A in enumerate(matrices, start=1):
        print(f"--- Тест {idx} (размерность {A.shape[0]}x{A.shape[1]}) ---")
        n = A.shape[0]

        # 1. Проверка симметричности
        assert np.allclose(A, A.T), f"Матрица {idx} не симметрична"

        # 2. Вызов метода вращений
        eigvals, V, iters, history = jacobi_rotation(A, eps=eps)

        # 3. Проверка ортогональности матрицы собственных векторов: V^T @ V == I
        assert np.allclose(V.T @ V, np.eye(n), atol=1e-5), (
            f"V не ортогональна на тесте {idx}"
        )
        print(f"✓ Ортогональность собственных векторов V проверена")

        # 4. Проверка невязки для каждого собственного вектора: ||A*v - lam*v|| < 1e-4
        for i in range(n):
            v_i = V[:, i]
            lam_i = eigvals[i]
            res = float(np.linalg.norm(A @ v_i - lam_i * v_i))
            assert res < 1e-4, f"Невязка для λ_{i + 1} слишком велика: {res}"

        print(f"✓ Невязки собственных пар A*v = λ*v проверены (шагов: {iters})")

        # 5. Сравнение собственных значений с numpy.linalg.eigh
        np_eigs = np.sort(np.linalg.eigh(A)[0])
        our_eigs = np.sort(eigvals)
        assert np.allclose(our_eigs, np_eigs, atol=1e-4), (
            f"Несовпадение собственных значений на тесте {idx}:\n"
            f"Получено:  {our_eigs}\nОжидалось: {np_eigs}"
        )
        print(f"✓ Собственные значения совпадают с эталоном numpy")
        print(f"  λ = {np.round(our_eigs, 4)}\n")

    print("Все тесты метода вращений Якоби успешно пройдены!\n")


if __name__ == "__main__":
    test_jacobi_rotation()

"""
Тестирование лабораторной работы 1.1 (LU-разложение).
"""

from pathlib import Path

import numpy as np

from lab1.io_utils import read_linear_systems
from lab1.lu import lu_det, lu_factor, lu_inverse, lu_solve


def test_lu_systems() -> None:
    test_file = Path(__file__).parent / "test_lu.txt"
    systems = read_linear_systems(test_file)

    print(
        f"\nЗапуск тестов LU-разложения из {test_file.name} (систем: {len(systems)})...\n"
    )

    for idx, (A, b) in enumerate(systems, start=1):
        print(f"--- Тест {idx} ---")
        n = A.shape[0]

        # 1. Факторизация PA = LU
        L, U, p, swaps = lu_factor(A)
        P_mat = np.eye(n)[p]
        assert np.allclose(P_mat @ A, L @ U), f"PA != LU на тесте {idx}"
        print("✓ Разложение PA = LU проверено")

        # 2. Решение СЛАУ Ax = b
        x = lu_solve(L, U, p, b)
        assert np.allclose(A @ x, b), f"Ax != b на тесте {idx}"
        print(f"✓ Решение найдено: x = {np.round(x, 4)}")

        # 3. Вычисление определителя
        det_custom = lu_det(U, swaps)
        det_numpy = float(np.linalg.det(A))
        assert np.isclose(det_custom, det_numpy), (
            f"Несовпадение det на тесте {idx}: {det_custom} != {det_numpy}"
        )
        print(f"✓ Определитель проверен: det(A) = {det_custom:.4f}")

        # 4. Обратная матрица
        A_inv = lu_inverse(L, U, p)
        assert np.allclose(A @ A_inv, np.eye(n)), f"A @ A^(-1) != I на тесте {idx}"
        print("✓ Обратная матрица проверена\n")

    print("Все тесты LU-разложения успешно пройдены!\n")


if __name__ == "__main__":
    test_lu_systems()

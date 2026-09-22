"""
Тестирование лабораторной работы 1.2 (Метод прогонки).
"""

from pathlib import Path

import numpy as np

from lab1.io_utils import read_tridiagonal_systems
from lab1.tridiagonal import (
    check_diagonal_dominance,
    compute_residual,
    tridiagonal_solve,
)


def test_tridiagonal_systems() -> None:
    test_file = Path(__file__).parent / "test_tridiagonal.txt"
    systems = read_tridiagonal_systems(test_file)

    print(
        f"\nЗапуск тестов метода прогонки из {test_file.name} (систем: {len(systems)})...\n"
    )

    for idx, (a, b, c, d) in enumerate(systems, start=1):
        print(f"--- Тест {idx} ---")

        # 1. Проверка диагонального преобладания
        is_dominant = check_diagonal_dominance(a, b, c)
        print(
            f"Диагональное преобладание: {'выполнено' if is_dominant else 'не выполнено'}"
        )

        # 2. Решение системы
        x = tridiagonal_solve(a, b, c, d)
        print(f"✓ Решение найдено: x = {np.round(x, 4)}")

        # 3. Проверка невязки
        residual_norm = compute_residual(a, b, c, d, x)
        assert np.isclose(residual_norm, 0.0, atol=1e-9), (
            f"Слишком большая невязка на тесте {idx}: {residual_norm}"
        )
        print(f"✓ Невязка ||Ax - d|| = {residual_norm:.4e}\n")

    print("Все тесты метода прогонки успешно пройдены!\n")


if __name__ == "__main__":
    test_tridiagonal_systems()

"""
Тестирование лабораторной работы 1.3 (Итерационные методы).
"""

from pathlib import Path

import numpy as np

from lab1.io_utils import read_linear_systems
from lab1.iterations import (
    prepare_iteration_system,
    seidel_iteration,
    simple_iteration,
)


def test_iteration_methods() -> None:
    test_file = Path(__file__).parent / "test_iterations.txt"
    systems = read_linear_systems(test_file)
    eps = 1e-6

    print(
        f"\nЗапуск тестов итерационных методов из {test_file.name} (eps = {eps})...\n"
    )

    for idx, (A, b) in enumerate(systems, start=1):
        print(f"--- Тест {idx} ---")

        # 1. Проверка условия сходимости
        _, _, alpha_norm = prepare_iteration_system(A, b)
        print(f"Норма ||alpha||_inf = {alpha_norm:.4f}")
        assert alpha_norm < 1.0, (
            f"Условие сходимости ||alpha|| < 1 нарушено на тесте {idx}"
        )

        # 2. Метод простых итераций (Якоби)
        x_simple, it_simple = simple_iteration(A, b, eps=eps)
        res_simple = float(np.max(np.abs(A @ x_simple - b)))
        assert res_simple < eps * 10, (
            f"Слишком большая невязка простых итераций: {res_simple}"
        )
        print(f"✓ Простые итерации: шагов = {it_simple}, невязка = {res_simple:.2e}")
        print(f"  x_simple = {np.round(x_simple, 4)}")

        # 3. Метод Зейделя
        x_seidel, it_seidel = seidel_iteration(A, b, eps=eps)
        res_seidel = float(np.max(np.abs(A @ x_seidel - b)))
        assert res_seidel < eps * 10, (
            f"Слишком большая невязка метода Зейделя: {res_seidel}"
        )
        print(f"✓ Метод Зейделя:    шагов = {it_seidel}, невязка = {res_seidel:.2e}")
        print(f"  x_seidel = {np.round(x_seidel, 4)}")

        # Метод Зейделя обычно сходится быстрее или за то же число шагов
        assert it_seidel <= it_simple, (
            "Метод Зейделя сделал больше шагов, чем метод простых итераций"
        )
        print(f"Статус: Тест {idx} успешно пройден!\n")

    print("Все тесты итерационных методов успешно пройдены!\n")


if __name__ == "__main__":
    test_iteration_methods()

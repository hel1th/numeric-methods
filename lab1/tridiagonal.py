from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from lab1.io_utils import read_tridiagonal_systems
except ImportError:
    from io_utils import read_tridiagonal_systems


def check_diagonal_dominance(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    n = len(b)
    has_strict = False

    if abs(b[0]) < abs(c[0]):
        return False
    if abs(b[0]) > abs(c[0]):
        has_strict = True

    for i in range(1, n - 1):
        row_sum = abs(a[i - 1]) + abs(c[i])
        if abs(b[i]) < row_sum:
            return False
        if abs(b[i]) > row_sum:
            has_strict = True

    if abs(b[n - 1]) < abs(a[n - 2]):
        return False
    if abs(b[n - 1]) > abs(a[n - 2]):
        has_strict = True

    return has_strict


def compute_residual(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
) -> float:
    n = len(d)
    Ax = np.zeros(n, dtype=np.float64)

    Ax[0] = b[0] * x[0] + c[0] * x[1]

    for i in range(1, n - 1):
        Ax[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]

    Ax[n - 1] = a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1]

    residual = Ax - d
    return float(np.linalg.norm(residual))


def tridiagonal_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    return_coeffs: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n = len(d)
    if len(b) != n or len(a) != n - 1 or len(c) != n - 1:
        raise ValueError("Несоответствие размерностей входных массивов.")

    P = np.zeros(n - 1, dtype=np.float64)
    Q = np.zeros(n, dtype=np.float64)

    if np.isclose(b[0], 0.0):
        raise ValueError("Деление на ноль: b[0] равен 0.")

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] + a[i - 1] * P[i - 1]
        if np.isclose(denom, 0.0):
            raise ValueError(f"Деление на ноль в знаменателе на шаге {i}.")

        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i - 1] * Q[i - 1]) / denom

    denom_last = b[n - 1] + a[n - 2] * P[n - 2]
    if np.isclose(denom_last, 0.0):
        raise ValueError(
            f"Деление на ноль в знаменателе на последнем шаге {n - 1}.")

    Q[n - 1] = (d[n - 1] - a[n - 2] * Q[n - 2]) / denom_last

    x = np.zeros(n, dtype=np.float64)
    x[n - 1] = Q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    if return_coeffs:
        return x, P, Q

    return x


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        __file__).parent / "data" / "tridiagonal.txt"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Файл с входными данными не найден: {data_file}")

    print("=" * 70)
    print("Лабораторная работа 1.2: Метод прогонки для трехдиагональных СЛАУ")
    print(f"Чтение входных данных из файла: {data_file}")
    print("=" * 70)

    systems = read_tridiagonal_systems(data_file)
    for idx, (a, b, c, d) in enumerate(systems, start=1):
        n = len(d)
        print(f"\n>>> СИСТЕМА {idx} (размерность n = {n}) <<<")
        print("Поддиагональ a (длина n-1):", a)
        print("Главная диагональ b (длина n):", b)
        print("Наддиагональ c (длина n-1):", c)
        print("Правая часть d (длина n):", d)

        is_dom = check_diagonal_dominance(a, b, c)
        print(
            f"Диагональное преобладание: {'выполнено' if is_dom else 'не выполнено'}")

        res = tridiagonal_solve(a, b, c, d, return_coeffs=True)
        assert isinstance(res, tuple)
        x, P, Q = res

        print("\nПрогоночные коэффициенты:")
        print("   Коэффициенты P (длина n-1):")
        for i, p_val in enumerate(P, start=1):
            print(f"     P_{i} = {p_val:12.6f}")
        print("   Коэффициенты Q (длина n):")
        for i, q_val in enumerate(Q, start=1):
            print(f"     Q_{i} = {q_val:12.6f}")

        print("\nИтоговое решение системы уравнений x:")
        for i, x_val in enumerate(x, start=1):
            print(f"     x_{i} = {x_val:12.6f}")

        residual = compute_residual(a, b, c, d, x)
        print(f"\n   Норма вектора невязки ||Ax - d||: {residual:.2e}")
        print("=" * 70)


if __name__ == "__main__":
    main()

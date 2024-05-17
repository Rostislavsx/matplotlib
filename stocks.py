import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import timeit

def sinc(x: float) -> float:
    """
    Calculate the sinc function value for a given x.
    """
    if x == 0:
        return 1.0
    return np.sin(x) / x

def func_py(x: list[float], N: int) -> list[float]:
    """
    Calculate function values for passed array of arguments.
    """
    return [sinc(2 * t / N - 1) for t in x]

def tabulate_py(a: float, b: float, n: int, N: int) -> dict[float]:
    x = [a + x * (b - a) / n for x in range(n)]
    y = func_py(x, N)
    return x, y

def tabulate_np(a: float, b: float, n: int, N: int) -> np.ndarray:
    x = np.linspace(a, b, n)
    y = np.sinc((2 * x / N - 1) / np.pi)  # numpy's sinc function is normalized, so we adjust accordingly
    return x, y

def plot_function_data(a: float, b: float, n: int, N: int) -> None:
    x_py, y_py = tabulate_py(a, b, n, N)
    x_np, y_np = tabulate_np(a, b, n, N)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_py, y_py, label='Python', alpha=0.6, color='cyan')
    plt.plot(x_np, y_np, label='NumPy', linestyle='dashed', alpha=0.6, color='magenta')
    
    plt.title(f"Function $f(x) = sinc(2x/N - 1)$ with N = {N}", fontsize=18)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

def main() -> None:
    a, b, N = 0, 1, 11  # Example parameters
    n_values = np.array((1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000), dtype="uint32")
    t_py = np.zeros_like(n_values, dtype=float)
    t_np = np.zeros_like(n_values, dtype=float)

    for i in range(len(n_values)):
        n = n_values[i]
        t_py[i] = 1_000_000 * timeit.timeit(f"tabulate_py(0, 1, {n}, {N})", number=100, globals=globals()) / 100
        t_np[i] = 1_000_000 * timeit.timeit(f"tabulate_np(0, 1, {n}, {N})", number=100, globals=globals()) / 100

    plt.plot(n_values, t_py / t_np)
    plt.xlabel("Number of points (n)")
    plt.ylabel("Execution time ratio (Python/NumPy)")
    plt.title("Execution Time Comparison: Python vs. NumPy")
    plt.grid()
    plt.show()

    # Plot function data for a given n
    plot_function_data(a, b, 1000, N)

if __name__ == "__main__":
    main()

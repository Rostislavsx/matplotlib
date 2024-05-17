import numpy as np
from matplotlib import pyplot as plt

def func_py(x: list[float], N: int) -> list[float]:
    """
    Calculate function values for passed array of arguments
    """
    return [ np.sin(2 * t * np.pi / N - np.pi / 2) if N % 2 == 1 else np.sin(t) for t in x ]

def tabulate_py(a: float, b: float, N: int) -> dict:
    x = [ a + x * (b - a) / N for x in range(N) ]
    y = func_py(x, N)  # Pass N to func_py
    return dict(zip(x, y))  # Return a dictionary

def tabulate_np(a: float, b: float, N: int) -> np.ndarray:
    x = np.linspace(a, b, N)
    y = func_py(x.tolist(), N)  # Pass N to func_py and convert x to list
    return np.array([x, y])  # Return a numpy array

def test_tabulation(f, a, b, N, axis):
    res = f(a, b, N)

    if isinstance(res, dict):  # Check if res is a dictionary
        axis.plot(list(res.keys()), list(res.values()))
    else:  # If res is a numpy array
        axis.plot(res[0], res[1])
    axis.grid()

def main():
    a, b = 0, 1
    N = 11

    fig, (ax1, ax2) = plt.subplots(2, 1)
    test_tabulation(tabulate_py, a, b, N, ax1)
    test_tabulation(tabulate_np, a, b, N, ax2)
    plt.show()

if __name__ == '__main__':
    main()
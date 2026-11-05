import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.integrate import quad
from scipy.special import airy

a: float = 0.6
b: float = 0.9
N: int = 10  # число конечных элементов

nodes: np.ndarray = np.linspace(a, b, N + 1)


def y_exact(x):
    """Аналитическое решение для сравнения в узлах."""

    def exact_at(xi):
        Ai, _, Bi, _ = airy(xi + 1)
        I1 = quad(lambda t: t**2 * np.exp(t) * airy(t + 1)[0], a, xi)[0]
        I2 = quad(lambda t: t**2 * np.exp(t) * airy(t + 1)[2], a, xi)[0]
        return np.exp(-xi) * (
            -8.030205725881711 * Ai
            + 0.675923144730242 * Bi
            + np.pi * (Bi * I1 - Ai * I2)
        )

    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return exact_at(float(x))
    return np.vectorize(exact_at, otypes=[float])(x)


def p(x: np.ndarray) -> np.ndarray:
    """Коэффициент при первой производной."""
    return np.full_like(x, 2, dtype=float)


def q(x: np.ndarray) -> np.ndarray:
    """Коэффициент при функции y."""
    return -x


def f(x: np.ndarray) -> np.ndarray:
    """Правая часть уравнения."""
    return np.square(x)


def solve_fem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Линейные элементы и метод Галёркина для краевой задачи."""
    size = len(x)
    K: np.ndarray = np.zeros((size, size), dtype=float)
    F: np.ndarray = np.zeros(size, dtype=float)

    # Двухточечная квадратура Гаусса точна для полиномов степени до трёх.
    gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])

    for element in range(size - 1):
        left, right = x[element], x[element + 1]
        h = right - left
        local_K = np.zeros((2, 2), dtype=float)
        local_F = np.zeros(2, dtype=float)
        derivatives = np.array([-1 / h, 1 / h])

        for point in gauss_points:
            xi = (left + right) / 2 + h * point / 2
            shape = np.array([(right - xi) / h, (xi - left) / h])
            weight = h / 2

            # Умноженное на -1 слабое уравнение:
            # int(y' v' - p y' v - q y v) + 2 y(b) v(b)
            #     = -int(f v) + 2 v(b) - 0.7 v(a).
            local_K += weight * (
                np.outer(derivatives, derivatives)
                - p(np.array([xi]))[0] * np.outer(shape, derivatives)
                - q(np.array([xi]))[0] * np.outer(shape, shape)
            )
            local_F -= weight * f(np.array([xi]))[0] * shape

        ids = np.array([element, element + 1])
        K[np.ix_(ids, ids)] += local_K
        F[ids] += local_F

    # y'(a)=0.7; 0.5 y'(b)+y(b)=1 => y'(b)=2-2 y(b).
    F[0] -= 0.7
    F[-1] += 2
    K[-1, -1] += 2

    solution = np.linalg.solve(K, F)
    residuals = F - K @ solution
    return solution, K, residuals


if __name__ == "__main__":
    result, K, residuals = solve_fem(nodes)
    exact = y_exact(nodes)
    errors = np.abs(result - exact)

    print("  k       x_k          y_k        y_exact      |ошибка|")
    for k, (xi, yi, truth, error) in enumerate(zip(nodes, result, exact, errors)):
        print(f"{k:3d}  {xi:8.5f}  {yi:12.8f}  {truth:12.8f}  {error:12.8f}")
    print(f"Максимальная погрешность в узлах: {errors.max():.8e}")
    print(f"Максимальная невязка системы: {np.abs(residuals).max():.8e}")
    print(f"Симметрична ли матрица: {np.allclose(K, K.T)}")

    print("\nПроверка при сгущении сетки:")
    for count in (10, 20, 40, 80):
        grid = np.linspace(a, b, count + 1)
        values, _, _ = solve_fem(grid)
        print(f"N={count:2d}, h={(b-a)/count:.6f}, ошибка={np.max(np.abs(values-y_exact(grid))):.8e}")

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 12,
         "axes.unicode_minus": False, "savefig.facecolor": "white"}
    )
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    blue, dark, gray = "#2563EB", "#172033", "#64748B"

    dense = np.linspace(a, b, 500)
    ax.plot(dense, y_exact(dense), color=gray, linestyle="--", linewidth=2,
            label="Аналитическое решение")
    ax.plot(nodes, result, color=blue, linewidth=3, solid_capstyle="round",
            label="Метод конечных элементов",
            path_effects=[pe.SimpleLineShadow(offset=(0, -2), alpha=0.12), pe.Normal()])
    ax.scatter(nodes, result, s=45, color=blue, edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_xlabel("$x$", fontsize=15, color=dark, labelpad=12)
    ax.set_ylabel("$y(x)$", fontsize=15, color=dark, labelpad=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(True, linestyle="--", linewidth=0.8, color="#CBD5E1", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD5E1")
    ax.tick_params(axis="both", colors=gray, labelsize=11, length=0, pad=9)
    ax.margins(x=0.03, y=0.10)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.14, top=0.92)
    fig.savefig("fem_solution.png", dpi=400, bbox_inches="tight", pad_inches=0.25)
    fig.savefig("fem_solution.pdf", bbox_inches="tight", pad_inches=0.25)
    plt.show()

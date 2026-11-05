import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from progonka_matrix import clean_solve_progonka_matrix

L: int = 5
T: int = 10
N: int = 40  # число шагов разбиения по x
M: int = 2000  # число шагов разбиения по t

h: float = L / N
tau: float = T / M

a: float = 10
b: float = 3.0
c: float = 2.0
alpha: np.ndarray = np.array([1, 1])
beta: np.ndarray = np.array([0, 0])


t: np.ndarray = np.array([j * tau for j in range(M + 1)])
x: np.ndarray = np.array([i * h for i in range(N + 1)])

u: np.ndarray = np.zeros((N + 1, M + 1), dtype=float)
A: np.ndarray = np.zeros(N + 1, dtype=float)
B: np.ndarray = np.zeros(N + 1, dtype=float)
C: np.ndarray = np.zeros(N + 1, dtype=float)
D: np.ndarray = np.zeros(N + 1, dtype=float)

progonka_matrix: np.ndarray = np.zeros((N + 1, N + 1), dtype=float)
progonka_vector: np.ndarray = np.zeros(N + 1, dtype=float)


def u_exact(x, t, terms=300):
    x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))

    xx, tt = x.ravel(), t.ravel()
    k = b / (2 * a)
    mu = np.arange(1, terms + 1) * np.pi / L

    # Коэффициенты полиномов для начальных условий
    # и неоднородной правой части
    coeff = np.array(
        [
            [0, 1 / L - 1, 0],
            [0, 0.2, -1 / (20 * L)],
            [-a / L + b, -b / L + c - 0.01, -(c - 0.01) / (2 * L)],
            [a / L, b / L, (c + 0.01) / (2 * L)],
        ]
    )

    # Точные интегралы от x^j * exp(z*x)
    def moments(z):
        ez = np.exp(z * L)
        j0 = np.expm1(z * L) / z
        j1 = (L * ez - j0) / z
        j2 = (L**2 * ez - 2 * j1) / z
        return np.array([j0, j1, j2])

    m0 = moments(2 * k)
    p0 = coeff @ m0 / m0[0]

    mn = moments(k + 1j * mu)
    pn = np.real((coeff @ mn) * (1 - 1j * k / mu)) / (L / 2 * (1 + (k / mu) ** 2))

    q0, v0, fe, fc = np.column_stack([p0, pn])

    lam = np.r_[c, c - b * b / (4 * a) - a * mu**2]

    ae = fe / (0.01 - lam)
    ac = fc / (-0.01 - lam)

    H = q0 - ae - ac
    V = v0 + 0.1 * ae

    q = ae[:, None] * np.exp(-tt[None, :] / 10) + ac[:, None] * np.cos(tt[None, :] / 10)

    w0 = np.sqrt(lam[0])
    q[0] += H[0] * np.cosh(w0 * tt) + V[0] * np.sinh(w0 * tt) / w0

    wn = np.sqrt(-lam[1:])
    q[1:] += (
        H[1:, None] * np.cos(wn[:, None] * tt)
        + V[1:, None] * np.sin(wn[:, None] * tt) / wn[:, None]
    )

    basis = np.exp(-k * xx)[None, :] * (
        np.cos(mu[:, None] * xx) + (k / mu)[:, None] * np.sin(mu[:, None] * xx)
    )

    P = xx - xx**2 / (2 * L)
    Q = xx**2 / (2 * L)

    result = (
        P * np.exp(-tt / 10)
        + Q * np.cos(tt / 10)
        + q[0]
        + np.sum(basis * q[1:], axis=0)
    )

    return result.reshape(x.shape)


def phi1(t: float) -> float:
    return np.exp(-t / 10)


def phi2(t: float) -> float:
    return np.cos(t / 10)


def psi1(x: float) -> float:
    return x / 5


def psi2(x: float) -> float:
    return x / 10


# первые два слоя
for i in range(N + 1):
    u[i, 0] = psi1(x[i])

for i in range(N + 1):
    u[i, 1] = u[i, 0] + tau * psi2(x[i])


for tj in range(1, M):
    # крайняя левая точка
    A[0] = 0
    B[0] = beta[0] * h - alpha[0]
    C[0] = alpha[0]
    D[0] = h * phi1(t[tj + 1])
    progonka_matrix[0, 0:2] = np.array([B[0], C[0]])
    progonka_vector[0] = D[0]

    # внутренние точки
    for i in range(1, N):
        A[i] = a * tau**2 / h**2 - b * tau**2 / (2 * h)
        # c * tau ** 2, если без тау в квадрате то результат не очень правдоподобный
        B[i] = c * tau**2 - 1 - 2 * a * tau**2 / h**2
        C[i] = a * tau**2 / h**2 + b * tau**2 / (2 * h)
        D[i] = -2 * u[i, tj] + u[i, tj - 1]
        progonka_matrix[i, i - 1 : i + 2] = np.array([A[i], B[i], C[i]])
        progonka_vector[i] = D[i]

    # крайняя правая точка
    A[N] = -alpha[1]
    B[N] = alpha[1] + h * beta[1]
    C[N] = 0
    D[N] = h * phi2(t[tj + 1])
    progonka_matrix[N, N - 1 : N + 1] = np.array([A[N], B[N]])
    progonka_vector[N] = D[N]

    u[:, tj + 1] = clean_solve_progonka_matrix(progonka_matrix, progonka_vector)[0]  # type: ignore


print(np.array([np.round(i, 2) for i in u]).T)

X, TT = np.meshgrid(x, t, indexing="ij")

u_true = u_exact(X, TT)

errors = np.abs(u - u_true)

print(f"Максимальная погрешность: {errors.max():.8e}")
print(f"Средняя погрешность: {errors.mean():.8e}")

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

# Одинаковый масштаб для сравнения численной и аналитической поверхностей.
scale_min = min(u.min(), u_true.min())
scale_max = max(u.max(), u_true.max())

ax.plot_surface(X[:, ::10], TT[:, ::10], u[:, ::10], cmap="viridis",
                vmin=scale_min, vmax=scale_max)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_zlim(scale_min, scale_max)

fig_exact = plt.figure()
ax_exact = fig_exact.add_subplot(111, projection="3d")
ax_exact.plot_surface(X[:, ::10], TT[:, ::10], u_true[:, ::10], cmap="viridis",
                      vmin=scale_min, vmax=scale_max)
ax_exact.set_xlabel("x")
ax_exact.set_ylabel("t")
ax_exact.set_zlim(scale_min, scale_max)
ax_exact.view_init(elev=ax.elev, azim=ax.azim)
fig_exact.set_size_inches(8, 6)
fig_exact.subplots_adjust(right=0.83)
fig_exact.text(0.87, 0.52, "u(x,t)", rotation=90, va="center")
fig_exact.savefig("analytical_surface2.png", dpi=300, bbox_inches="tight")

# Наложение поверхностей на одни оси.
fig_overlay = plt.figure(figsize=(12, 5.5), dpi=180)
ax_overlay = fig_overlay.add_subplot(121, projection="3d")
ax_overlay_error = fig_overlay.add_subplot(122)

ax_overlay.plot_surface(X[:, ::10], TT[:, ::10], u[:, ::10],
                        color="#2563EB", alpha=0.48, linewidth=0)
ax_overlay.plot_wireframe(X[:, ::10], TT[:, ::10], u_true[:, ::10],
                          rstride=4, cstride=16, color="#EA580C", linewidth=0.85)
ax_overlay.set_xlabel("x")
ax_overlay.set_ylabel("t")
ax_overlay.set_zlabel("u(x,t)", labelpad=0)
ax_overlay.set_zlim(scale_min, scale_max)
ax_overlay.view_init(elev=ax.elev, azim=ax.azim)
ax_overlay.legend(
    handles=[Patch(color="#2563EB", alpha=0.48, label="Численное решение"),
             Line2D([], [], color="#EA580C", label="Аналитическое решение")],
    loc="upper left", fontsize=9, frameon=False,
)
ax_overlay_error.plot(t, errors.max(axis=0), color="#172033", linewidth=2)
ax_overlay_error.set_xlabel("t")
ax_overlay_error.set_ylabel(r"$\max_x |u(x,t)-u_{\mathrm{an}}(x,t)|$")
ax_overlay_error.ticklabel_format(axis="y", style="sci", scilimits=(4, 4))
ax_overlay_error.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
ax_overlay_error.spines["top"].set_visible(False)
ax_overlay_error.spines["right"].set_visible(False)
fig_overlay.subplots_adjust(left=0.03, right=0.95, bottom=0.14, top=0.94, wspace=0.35)
fig_overlay.savefig("surface_comparison2.png", dpi=300, bbox_inches="tight", pad_inches=0.18)

# На общей 3D-шкале разница плохо видна: сравним сечения при t = T.
fig_compare, ax_profile = plt.subplots(figsize=(7, 4.5), dpi=180)
ax_profile.plot(x, u[:, -1] / 1e6, color="#2563EB", linewidth=2.5,
                label="Численное решение")
ax_profile.plot(x, u_true[:, -1] / 1e6, color="#EA580C", linewidth=2.5,
                linestyle="--", label="Аналитическое решение")
ax_profile.fill_between(x, u[:, -1] / 1e6, u_true[:, -1] / 1e6,
                        color="#2563EB", alpha=0.08)
ax_profile.set_xlabel("x")
ax_profile.set_ylabel(r"$u(x,10)$, $10^6$")
ax_profile.set_ylim(u_true[:, -1].min() / 1e6 - 0.015,
                    u[:, -1].max() / 1e6 + 0.015)
ax_profile.legend(frameon=False, fontsize=9)

ax_profile.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
ax_profile.spines["top"].set_visible(False)
ax_profile.spines["right"].set_visible(False)
fig_compare.tight_layout()
fig_compare.savefig("solution_comparison2.png", dpi=300, bbox_inches="tight", pad_inches=0.18)

plt.show()

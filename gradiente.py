

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INICIAL = "00001001000011001100111"
ALVO = "00000000000000000000000"

RAIZ = Path(__file__).resolve().parent
FIGS = RAIZ / "figs"


AZUL, LARANJA, AQUA, VERM = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"
DIV_MIN, DIV_MEIO, DIV_MAX = "#2a78d6", "#f0efec", "#e34948"
CMAP_DIV = LinearSegmentedColormap.from_list("div_azul_verm", [DIV_MIN, DIV_MEIO, DIV_MAX])


def _estilo():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10.5,
            "axes.edgecolor": EIXO,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def bits_para_vetor(s: str) -> np.ndarray:
    return np.array(list(s), dtype=float)


def L_erro(x: np.ndarray, y: np.ndarray) -> float:

    return float(np.sum((x - y) ** 2))


def grad_L(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * (x - y)


def gradiente_descendente(x0, y, eta=0.25, iters=500, tol=1e-12):


    x = np.array(x0, dtype=float)
    xs, dxs = [x.copy()], []
    for _ in range(iters):
        dx = -eta * grad_L(x, y)
        x = x + dx
        xs.append(x.copy())
        dxs.append(dx)
        if L_erro(x, y) < tol:
            break
    return np.array(xs), np.array(dxs)


def estados_discretos(xs: np.ndarray) -> list[str]:

    return ["".join("1" if v >= 0.5 else "0" for v in linha) for linha in xs]


def z_de_x(x, y, eta):


    return -eta * grad_L(x, y)


def custo_C(z):

    d = z[:-1] - z[1:]
    return float(np.sum(d * d))


def grad_C_z(z):


    dif = z[:-1] - z[1:]
    g = np.zeros_like(z)
    g[:-1] += 2.0 * dif
    g[1:] -= 2.0 * dif
    return g


def custo_J(x, y, eta, lam):
    return L_erro(x, y) + lam * custo_C(z_de_x(x, y, eta))


def grad_J(x, y, eta, lam):


    return grad_L(x, y) - 2.0 * eta * lam * grad_C_z(z_de_x(x, y, eta))


def minimiza_J(x0, y, eta=0.1, lam=1.0, iters=500, tol=1e-12):


    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    for _ in range(iters):
        x = x - eta * grad_J(x, y, eta, lam)
        xs.append(x.copy())
        if not np.isfinite(x).all() or custo_J(x, y, eta, lam) > 1e12:
            break
        if custo_J(x, y, eta, lam) < tol:
            break
    return np.array(xs)


def _grad_numerico(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def testes() -> None:
    rng = np.random.default_rng(0)

    for _ in range(20):
        L = int(rng.integers(2, 12))
        x = rng.uniform(0, 1, L)
        y = rng.integers(0, 2, L).astype(float)
        eta = rng.uniform(0.05, 0.45)
        lam = rng.uniform(0.0, 5.0)
        assert np.allclose(grad_L(x, y), _grad_numerico(lambda v: L_erro(v, y), x))
        assert np.allclose(
            grad_C_z(x), _grad_numerico(custo_C, x), atol=1e-5
        )
        assert np.allclose(
            grad_J(x, y, eta, lam),
            _grad_numerico(lambda v: custo_J(v, y, eta, lam), x),
            atol=1e-4,
        )

    x0, y = bits_para_vetor(INICIAL), bits_para_vetor(ALVO)

    xs, dxs = gradiente_descendente(x0, y, eta=0.25, iters=200)
    assert L_erro(xs[-1], y) < 1e-10
    assert xs.min() >= -1e-12 and xs.max() <= 1 + 1e-12

    xs, _ = gradiente_descendente(x0, y, eta=0.6, iters=10)
    assert xs[:, 4].min() < 0, "esperado x_k < 0 para η > 1/2"

    assert np.isclose(xs[1, 4], (1 - 1.2) * 1.0)

    xs, dxs = gradiente_descendente(x0, y, eta=0.3, iters=5)
    assert np.allclose(dxs[0], z_de_x(x0, y, 0.3))
    print("testes parte 2: ok")


def demo_eta() -> None:

    _estilo()
    FIGS.mkdir(exist_ok=True)
    x0, y = bits_para_vetor(INICIAL), bits_para_vetor(ALVO)
    k = INICIAL.index("1")
    etas = [0.25, 0.4, 0.6, 0.75]
    cores = [AZUL, AQUA, LARANJA, VERM]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for eta, cor in zip(etas, cores):
        xs, _ = gradiente_descendente(x0, y, eta=eta, iters=40)
        ax1.plot(xs[:, k], color=cor, lw=2, label=f"η = {eta}")
        ax2.semilogy([max(L_erro(linha, y), 1e-300) for linha in xs], color=cor, lw=2)
        if eta > 0.5:
            assert xs[:, k].min() < 0
    ax1.axhspan(0, 1, color=GRADE, alpha=0.4)
    ax1.axhline(0, color=MUTED, lw=1)
    ax1.set_xlabel("iteração t")
    ax1.set_ylabel(f"x_{k}(t)")
    ax1.set_title("bit que muda de 1 → 0")
    ax1.legend()
    ax2.set_xlabel("iteração t")
    ax2.set_ylabel("L(x_t)")
    ax2.set_title("convergência do erro")
    fig.suptitle("η ≤ 1/2 converge dentro de [0, 1]; η > 1/2 extrapola (x_k < 0)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "parte2_eta.png", bbox_inches="tight")
    plt.close(fig)


def demo_transformacoes() -> None:

    x0, y = bits_para_vetor(INICIAL), bits_para_vetor(ALVO)
    xs, dxs = gradiente_descendente(x0, y, eta=0.25, iters=100)
    disc = estados_discretos(xs)
    trocas = [t for t in range(1, len(disc)) if disc[t] != disc[t - 1]]
    print(f"  passos: {len(dxs)}; ‖Δx_0‖ = {np.linalg.norm(dxs[0]):.3f} "
          f"(decai ~|1−2η|^t = {abs(1 - 0.5):.2f} por passo)")
    print(f"  estados discretos visitados: {len(set(disc))}")
    for t in trocas:
        mudam = [k for k in range(len(INICIAL)) if disc[t][k] != disc[t - 1][k]]
        print(f"  t = {t}: {len(mudam)} bits cruzam 1/2 (posições {mudam})")
    print(f"  estado final == alvo: {disc[-1] == ALVO}")


def demo_lambda() -> None:


    _estilo()
    FIGS.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    L = 32
    x0 = rng.uniform(0, 1, L)
    y = rng.integers(0, 2, L).astype(float)
    eta = 0.1
    lam_max = (1 - eta) / (16 * eta**3)
    lams = [0.0, 1.0, 10.0, 50.0, 100.0]
    iters = 300
    traj = {}

    print(f"\nMinimizando J(x) para η = {eta} fixo, L = {L}, x0 fuzzy aleatório")
    print(f"(limite de estabilidade: λ_max = (1−η)/(16η³) ≈ {lam_max:.1f})")
    print(f"{'λ':>6} {'L(x_300)':>10} {'L(x_15)':>10} {'Σ C(z_t)':>10} "
          f"{'t*':>4} {'min x_k':>9} {'max x_k':>9}")
    for lam in lams:
        xs = minimiza_J(x0, y, eta=eta, lam=lam, iters=iters)
        traj[lam] = xs


        zs = z_de_x(xs[:-1], y, eta)
        soma_C = sum(custo_C(z) for z in zs)
        limiar = (xs >= 0.5).astype(int)
        t_estrela = next(
            (t for t in range(len(xs)) if np.array_equal(limiar[t], y.astype(int))),
            None,
        )
        L_fim = L_erro(xs[-1], y)
        L15 = L_erro(xs[min(15, len(xs) - 1)], y)
        print(f"{lam:>6g} {L_fim:>10.2e} {L15:>10.2e} {soma_C:>10.2e} "
              f"{str(t_estrela):>4} {xs.min():>9.3f} {xs.max():>9.3f}"
              + ("   <- diverge" if L_fim > 1e6 or not np.isfinite(L_fim) else ""))
    print("(Σ C(z_t) soma o custo sobre os pesos z_t = −η∇L(x_t), o termo λC(z)"
          " de J ao longo da trajetória)")


    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lam, cor in zip(lams, [MUTED, AQUA, AZUL, LARANJA, VERM]):
        perdas = [min(L_erro(linha, y), 1e6) for linha in traj[lam]]
        ax.semilogy(perdas, color=cor, lw=2, label=f"λ = {lam:g}")
    ax.axhline(1e6, color=GRADE, lw=1)
    ax.annotate("λ = 100 > λ_max ≈ 56: instável", xy=(150, 1e5), color=INK2,
                fontsize=9)
    ax.set_xlabel("iteração t")
    ax.set_ylabel("L(x_t)")
    ax.set_title("Erro por iteração: λ coordena vizinhos e pode instabilizar")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "parte2_lambda_perda.png", bbox_inches="tight")
    plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, lam in zip(axes, [0.0, 50.0]):
        xs = traj[lam]
        norm = TwoSlopeNorm(vcenter=0.5, vmin=min(-0.2, xs.min()), vmax=1.0)
        im = ax.imshow(
            xs.T, aspect="auto", origin="lower", norm=norm, cmap=CMAP_DIV,
            interpolation="nearest",
        )
        ax.set_xlabel("iteração t")
        ax.set_title(f"λ = {lam:g}")
        ax.grid(False)
    axes[0].set_ylabel("bit k")
    fig.colorbar(im, ax=axes, label="x_k(t)", shrink=0.9)
    fig.suptitle("Trajetória x_k(t): com λ = 50 as transformações acoplam vizinhos",
                 y=1.00)
    fig.savefig(FIGS / "parte2_lambda_mapa.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    testes()
    print("\nSequência de transformações (II.C, η = 0.25, sem custo):")
    demo_transformacoes()
    demo_eta()
    print("\nFigura: figs/parte2_eta.png")
    demo_lambda()
    print("Figuras: figs/parte2_lambda_perda.png, figs/parte2_lambda_mapa.png")


if __name__ == "__main__":
    main()

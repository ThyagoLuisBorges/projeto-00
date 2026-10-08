

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INICIAL = "00001001000011001100111"
ALVO = "00000000000000000000000"

RAIZ = Path(__file__).resolve().parent
FIGS = RAIZ / "figs"
DADOS = RAIZ / "dados"


AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"


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


def flip_bit(s: str, i: int) -> str:

    return s[:i] + ("1" if s[i] == "0" else "0") + s[i + 1 :]


def flip_intervalo(s: str, i: int, j: int) -> str:

    trecho = "".join("1" if b == "0" else "0" for b in s[i : j + 1])
    return s[:i] + trecho + s[j + 1 :]


def aplicar(s: str, op: tuple[int, int]) -> str:

    return flip_intervalo(s, op[0], op[1])


def operacoes(L: int) -> list[tuple[int, int]]:

    return [(i, j) for i in range(L) for j in range(i, L)]


def _bits(s: str) -> np.ndarray:

    return np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")


def _blocos(d: np.ndarray) -> list[tuple[int, int]]:

    pad = np.concatenate(([False], d, [False]))
    bordas = np.flatnonzero(pad[1:] != pad[:-1])
    return [(int(a), int(b - 1)) for a, b in zip(bordas[0::2], bordas[1::2])]


def blocos_diferentes(s: str, alvo: str) -> list[tuple[int, int]]:

    return _blocos(_bits(s) != _bits(alvo))


def h_blocos(s: str, alvo: str) -> int:


    return len(blocos_diferentes(s, alvo))


def bfs(inicial: str, alvo: str, max_nos: int | None = None):


    L = len(inicial)
    assert len(alvo) == L
    t0 = time.perf_counter()
    if inicial == alvo:
        return [], 0, {"estados_expandidos": 0, "ops_geradas": 0, "tempo_s": 0.0}
    ops = operacoes(L)

    mascaras = [((1 << (j - i + 1)) - 1) << (L - 1 - j) for i, j in ops]
    s0, sT = int(inicial, 2), int(alvo, 2)
    pai = {s0: None}
    fila = deque([s0])
    expandidos = geradas = 0
    while fila:
        s = fila.popleft()
        expandidos += 1
        if max_nos is not None and expandidos > max_nos:
            return None, None, {"estados_expandidos": expandidos, "ops_geradas": geradas,
                                "tempo_s": time.perf_counter() - t0}
        for k, m in enumerate(mascaras):
            t = s ^ m
            if t in pai:
                continue
            geradas += 1
            pai[t] = (s, k)
            if t == sT:
                caminho = []
                no = sT
                while pai[no] is not None:
                    anterior, k2 = pai[no]
                    caminho.append(ops[k2])
                    no = anterior
                caminho.reverse()
                return caminho, len(caminho), {
                    "estados_expandidos": expandidos,
                    "ops_geradas": geradas,
                    "tempo_s": time.perf_counter() - t0,
                }
            fila.append(t)
    raise RuntimeError("alvo inalcançável (não deveria acontecer)")


def gulosa(inicial: str, alvo: str):


    t0 = time.perf_counter()
    x = _bits(inicial).copy()
    y = _bits(alvo)
    seq: list[tuple[int, int]] = []
    while True:
        blocos = _blocos(x != y)
        if not blocos:
            break
        i, j = max(blocos, key=lambda ij: ij[1] - ij[0])
        x[i : j + 1] ^= 1
        seq.append((i, j))
    return seq, len(seq), {"passos": len(seq), "tempo_s": time.perf_counter() - t0}


def _gulosa_todas_ops(inicial: str, alvo: str) -> list[tuple[int, int]]:


    s, seq = inicial, []
    while s != alvo:
        cand = [(h_blocos(aplicar(s, op), alvo), -(op[1] - op[0]), op) for op in operacoes(len(s))]
        _, _, (i, j) = min(cand)
        s = aplicar(s, (i, j))
        seq.append((i, j))
    return seq


TAMANHOS = [2**n for n in range(1, 13)]


def _par_aleatorio(rng: np.random.Generator, L: int) -> tuple[str, str]:
    a = rng.integers(0, 2, L)
    b = rng.integers(0, 2, L)
    f = lambda v: "".join(map(str, v.tolist()))
    return f(a), f(b)


def benchmark(N: int = 10, bfs_max_L: int = 16, seed: int = 42) -> list[dict]:


    rng = np.random.default_rng(seed)
    linhas = []
    for L in TAMANHOS:
        for par in range(N):
            a, b = _par_aleatorio(rng, L)
            _, custo_g, stg = gulosa(a, b)
            linha = {
                "L": L,
                "par": par,
                "custo_gulosa": custo_g,
                "tempo_gulosa_s": stg["tempo_s"],
                "estados_gulosa": stg["passos"],
            }
            if L <= bfs_max_L:
                _, custo_b, stb = bfs(a, b, max_nos=2_000_000)
                linha.update(
                    {
                        "custo_bfs": custo_b,
                        "tempo_bfs_s": stb["tempo_s"],
                        "estados_expandidos_bfs": stb["estados_expandidos"],
                        "ops_geradas_bfs": stb["ops_geradas"],
                    }
                )
            linhas.append(linha)
        print(f"  L = {L:5d} concluído")
    DADOS.mkdir(exist_ok=True)
    campos = list(linhas[0].keys())
    with open(DADOS / "benchmark.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    return linhas


def _media_desvio(linhas: list[dict], chave: str):
    por_L: dict[int, list] = {}
    for ln in linhas:
        if ln.get(chave) is not None:
            por_L.setdefault(ln["L"], []).append(ln[chave])
    Ls = sorted(por_L)
    return (
        Ls,
        [float(np.mean(por_L[L])) for L in Ls],
        [float(np.std(por_L[L])) for L in Ls],
    )


def graficos(linhas: list[dict]) -> None:

    _estilo()
    FIGS.mkdir(exist_ok=True)


    Lg, cg, dg = _media_desvio(linhas, "custo_gulosa")
    Lb, cb, db = _media_desvio(linhas, "custo_bfs")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))


    ax.plot(Lb, cb, "-s", color=AZUL, ms=8, lw=2, mfc="none", label="BFS (≤ 16)")
    ax.plot(Lg, cg, "-o", color=LARANJA, ms=5, lw=2, label="Gulosa")
    ax.plot(Lg, [(L + 1) / 4 for L in Lg], "--", color=MUTED, lw=1.2,
            label="teórico (L+1)/4")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("L (comprimento da sequência)")
    ax.set_ylabel("custo médio da solução")
    ax.set_title("Custo da solução: BFS × Gulosa (média de pares aleatórios)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "parte1_custo.png", bbox_inches="tight")
    plt.close(fig)


    Lg, tg, _ = _media_desvio(linhas, "tempo_gulosa_s")
    Lb, tb, _ = _media_desvio(linhas, "tempo_bfs_s")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(Lg, tg, "-o", color=LARANJA, ms=5, lw=2, label="Gulosa (todos os L)")
    ax.plot(Lb, tb, "-s", color=AZUL, ms=6, lw=2, label="BFS (viável só até L = 16)")
    ax.axvspan(16, 4096, color=GRADE, alpha=0.45)
    ax.annotate("BFS inviável: espaço de estados 2^L", xy=(40, 3e-6), color=INK2,
                fontsize=9, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("L (comprimento da sequência)")
    ax.set_ylabel("tempo médio por par (s)")
    ax.set_title("Tempo de busca: BFS explode com L; gulosa cresce com $L^2$")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(FIGS / "parte1_tempo.png", bbox_inches="tight")
    plt.close(fig)


    Lb, eb, _ = _media_desvio(linhas, "estados_expandidos_bfs")
    _, gb, _ = _media_desvio(linhas, "ops_geradas_bfs")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(Lb, eb, "-s", color=AZUL, ms=6, lw=2, label="estados expandidos")
    ax.plot(Lb, gb, "-^", color=AQUA, ms=6, lw=2, label="operações geradas")
    ax.plot(Lb, [2.0**L for L in Lb], ":", color=MUTED, lw=1.4, label="2^L (espaço total)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("L (comprimento da sequência)")
    ax.set_ylabel("esforço de busca (BFS)")
    ax.set_title("Esforço da BFS cresce como o espaço 2^L")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "parte1_esforco.png", bbox_inches="tight")
    plt.close(fig)


def testes() -> None:
    assert flip_bit("00", 0) == "10"
    assert flip_bit("01", 1) == "00"
    assert flip_intervalo("0000", 1, 2) == "0110"
    assert flip_intervalo("0000", 0, 3) == "1111"
    assert flip_intervalo("1010", 1, 1) == "1110" and flip_bit("1010", 1) == "1110"


    assert h_blocos(INICIAL, ALVO) == 5

    rng = np.random.default_rng(0)
    for _ in range(40):
        L = int(rng.integers(1, 11))
        a, b = _par_aleatorio(rng, L)
        for nome, (ops, custo, _) in (("bfs", bfs(a, b)), ("gulosa", gulosa(a, b))):
            s = a
            for op in ops:
                assert 0 <= op[0] <= op[1] < L
                s = aplicar(s, op)
            assert s == b, f"{nome} não chega ao alvo"
            assert custo == h_blocos(a, b), f"{nome} não é ótimo"


    for _ in range(10):
        L = int(rng.integers(1, 8))
        a, b = _par_aleatorio(rng, L)
        assert len(_gulosa_todas_ops(a, b)) == gulosa(a, b)[1]
    print("testes parte 1: ok")


def _fmt_op(op: tuple[int, int]) -> str:
    i, j = op
    return f"flip[{i}..{j}]" if i != j else f"flip[{i}]"


def main() -> None:
    testes()

    print("\nExemplo do enunciado (L = 23)")
    print(f"  inicial: {INICIAL}")
    print(f"  alvo   : {ALVO}")
    ops_b, custo_b, stb = bfs(INICIAL, ALVO, max_nos=2_000_000)
    print(f"  BFS    : custo {custo_b} | " + " ".join(map(_fmt_op, ops_b)))
    print(f"          ({stb['estados_expandidos']} estados expandidos, "
          f"{stb['ops_geradas']} operações geradas)")
    ops_g, custo_g, stg = gulosa(INICIAL, ALVO)
    print(f"  Gulosa : custo {custo_g} | " + " ".join(map(_fmt_op, ops_g)))

    print(f"\nBenchmark: N pares aleatórios para L = 2, 4, ..., 4096")
    linhas = benchmark()
    graficos(linhas)

    print("\nResumo (médias por L):")
    print(f"{'L':>5} {'custo_gulosa':>13} {'custo_bfs':>10} "
          f"{'tempo_gulosa_s':>15} {'tempo_bfs_s':>12} {'est_exp_bfs':>11}")
    for L in TAMANHOS:
        sub = [ln for ln in linhas if ln["L"] == L]

        def media(chave):
            vals = [ln[chave] for ln in sub if ln.get(chave) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        print(f"{L:>5} {media('custo_gulosa'):>13.2f} {media('custo_bfs'):>10.2f} "
              f"{media('tempo_gulosa_s'):>15.3e} {media('tempo_bfs_s'):>12.3e} "
              f"{media('estados_expandidos_bfs'):>11.0f}")
    print(f"\nSaídas: {DADOS/'benchmark.csv'} e figs/parte1_*.png")


if __name__ == "__main__":
    main()

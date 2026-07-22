"""Step 2 — Commonality Analysis 민감도 분석.

헤드라인 질문(§11-3, 문서 25 §3 Step 2):
    "equipment_correlation이 몇 이상이면 commonality가 원인 장비를 top-1으로 짚는가?"

탐색 결과 실제 지배 변수는 correlation이 아니라 **defect 표본 크기(= lot/경로 다양성)**
였다. 그래서 단일 축이 아니라 (correlation × n_lots) 2D 격자로 측정한다.

핵심 메커니즘 — 왜 표본이 지배하는가:
    commonality는 "불량이 특정 장비에 집중되는가"를 lift로 잡는다. 그런데 한 lot(25장)이
    거친 6개 장비는 그 lot의 불량과 **함께** 등장한다(교락). lot이 적으면 참 장비가
    공범 5개와 구별되지 않아 top-1이 1/6 확률의 제비뽑기가 된다. lot이 많아지면 각
    affected lot의 공범 장비가 매번 달라져(경로 다양성) 참 장비만 일관되게 남아 분리된다.

비교 가능 조건(전 셀 고정):
    - 원인 장비 = ETCH-07, defect = Edge-Ring
    - 이상 상시 활성(지속시간 고정), ramp 0
    - wafers_per_lot = 25, background_defect_rate = 0.01(코드 상수)
    - 셀마다 seed 0..N-1 반복
결과에 각 셀의 평균 불량 표본 수를 함께 명기한다.

실행: python -m experiments.commonality_sensitivity  (producer/ 에서)
      또는 python experiments/commonality_sensitivity.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# producer/ 를 import 경로에 추가 (스크립트 직접 실행 대비)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fab_simulator import (  # noqa: E402
    FabSimulator,
    NORMAL_LABEL,
    commonality_analysis,
    commonality_logistic,
)

CAUSE_TOOL = "ETCH-07"
DEFECT = "Edge-Ring"


@dataclass
class Cell:
    correlation: float
    n_lots: int
    top1_rate: float  # 참 장비를 top-1으로 지목한 시드 비율
    mean_rank: float  # 참 장비의 평균 순위 (1이 최선)
    lift_mean: float  # 참 장비 lift 평균
    lift_std: float
    defects_mean: float  # 평균 불량 표본 수
    n_seeds: int


@dataclass
class MethodCell:
    """방법 비교용 — (correlation, n_lots) 셀에서 세 방법의 top-1 성공률."""

    correlation: float
    n_lots: int
    top1_lift: float  # lift 순위 (기존)
    top1_fisher: float  # Fisher p-value 오름차순 순위
    top1_logit: float  # 로지스틱 계수 내림차순 순위
    sig_rate_true: float  # 참 장비 p<0.01 비율 (Fisher 검출력)
    defects_mean: float
    n_seeds: int


def run_cell(correlation: float, n_lots: int, n_seeds: int) -> Cell:
    hits, ranks, lifts, defects = [], [], [], []
    for seed in range(n_seeds):
        # 에피소드 스트림은 이 실험에 불필요 → n_lots=1로 최소화(commonality는 process_lots 경로)
        sim = FabSimulator(seed=seed, n_lots=1)
        sim.schedule_anomaly(CAUSE_TOOL, DEFECT, start_tick=0, end_tick=10**9, ramp_ticks=0)
        sim.set_correlation(DEFECT, correlation)
        results = sim.process_lots(n_lots=n_lots, at_tick=100)

        defects.append(sum(1 for r in results if r.label != NORMAL_LABEL))
        comm = commonality_analysis(results, top_k=999)  # 전체 순위
        order = [c.tool_id for c in comm]
        if CAUSE_TOOL in order:
            rank = order.index(CAUSE_TOOL) + 1
            lift = next(c.lift for c in comm if c.tool_id == CAUSE_TOOL)
        else:
            rank, lift = len(order) + 1, 0.0  # 불량 0 등으로 미등장
        ranks.append(rank)
        lifts.append(lift)
        hits.append(1 if rank == 1 else 0)

    return Cell(
        correlation=correlation,
        n_lots=n_lots,
        top1_rate=float(np.mean(hits)),
        mean_rank=float(np.mean(ranks)),
        lift_mean=float(np.mean(lifts)),
        lift_std=float(np.std(lifts)),
        defects_mean=float(np.mean(defects)),
        n_seeds=n_seeds,
    )


def run_grid(
    correlations=(0.9, 0.7, 0.5, 0.3, 0.1),
    lot_counts=(20, 40, 80, 160),
    n_seeds: int = 24,
) -> list[Cell]:
    return [
        run_cell(c, n, n_seeds)
        for c in correlations
        for n in lot_counts
    ]


def print_grid(cells: list[Cell], correlations, lot_counts) -> None:
    idx = {(c.correlation, c.n_lots): c for c in cells}

    print("=" * 78)
    print("Step 2 — Commonality 민감도: top-1 성공률 (원인=ETCH-07, Edge-Ring)")
    print(f"          seed {cells[0].n_seeds}회 반복, 이상 상시 활성, wafers/lot=25")
    print("=" * 78)

    print("\n[A] top-1 성공률  (행=correlation, 열=n_lots)")
    header = "corr \\ n_lots  " + "".join(f"{n:>8}" for n in lot_counts)
    print(header)
    print("-" * len(header))
    for c in correlations:
        row = f"{c:<14}"
        for n in lot_counts:
            row += f"{idx[(c, n)].top1_rate:>8.2f}"
        print(row)

    print("\n[B] 참 장비 평균 순위  (1=완벽, 낮을수록 좋음)")
    print(header)
    print("-" * len(header))
    for c in correlations:
        row = f"{c:<14}"
        for n in lot_counts:
            row += f"{idx[(c, n)].mean_rank:>8.2f}"
        print(row)

    print("\n[C] 참 장비 lift 평균  (신호 세기; 1이면 무신호)")
    print(header)
    print("-" * len(header))
    for c in correlations:
        row = f"{c:<14}"
        for n in lot_counts:
            row += f"{idx[(c, n)].lift_mean:>8.2f}"
        print(row)

    print("\n[D] 평균 불량 표본 수  (셀 난이도의 실제 척도)")
    print(header)
    print("-" * len(header))
    for c in correlations:
        row = f"{c:<14}"
        for n in lot_counts:
            row += f"{idx[(c, n)].defects_mean:>8.0f}"
        print(row)


def summarize(cells: list[Cell], lot_counts, correlations) -> None:
    idx = {(c.correlation, c.n_lots): c for c in cells}
    print("\n" + "=" * 78)
    print("요약 — 지배 변수 판별")
    print("=" * 78)

    # 표본 축 효과: corr 고정(0.5), n_lots에 따른 top-1
    print("\n· 표본(n_lots) 축 효과 @ corr=0.5:")
    for n in lot_counts:
        c = idx.get((0.5, n))
        if c:
            print(f"    n_lots={n:>3}: top-1={c.top1_rate:.2f}, 참장비 순위={c.mean_rank:.2f}, 불량~{c.defects_mean:.0f}장")

    # correlation 축 효과: n_lots 고정(40 = 전이 구간), corr에 따른 top-1
    print("\n· correlation 축 효과 @ n_lots=40(전이 구간):")
    for c in correlations:
        cell = idx.get((c, 40))
        if cell:
            print(f"    corr={c}: top-1={cell.top1_rate:.2f}, lift={cell.lift_mean:.2f}")

    # 최소 표본 항 = 교락 바닥 (1/6 근처)
    tiny = [idx[(c, lot_counts[0])].top1_rate for c in correlations if (c, lot_counts[0]) in idx]
    if tiny:
        print(f"\n· 최소 표본(n_lots={lot_counts[0]}) top-1 평균 = {np.mean(tiny):.2f}  "
              f"(≈1/6=0.17이면 6개 공범 장비와 교락된 제비뽑기 바닥)")

    # 운영선: top-1 >= 0.9 되는 최소 n_lots (corr별)
    print("\n· 운영선 — top-1 ≥ 0.90 되는 최소 n_lots:")
    for c in correlations:
        need = None
        for n in lot_counts:
            if idx[(c, n)].top1_rate >= 0.90:
                need = n
                break
        print(f"    corr={c}: n_lots ≥ {need if need else '>'+str(lot_counts[-1])}")


def run_method_cell(correlation: float, n_lots: int, n_seeds: int) -> MethodCell:
    """한 (correlation, n_lots) 셀에서 3-방법(lift/Fisher/logistic)의 top-1을 잰다."""
    h_lift, h_fisher, h_logit, sig_true, defects = [], [], [], [], []
    for seed in range(n_seeds):
        sim = FabSimulator(seed=seed, n_lots=1)
        sim.schedule_anomaly(CAUSE_TOOL, DEFECT, start_tick=0, end_tick=10**9, ramp_ticks=0)
        sim.set_correlation(DEFECT, correlation)
        results = sim.process_lots(n_lots=n_lots, at_tick=100)
        defects.append(sum(1 for r in results if r.label != NORMAL_LABEL))

        comm = commonality_analysis(results, top_k=999)
        if not comm:
            h_lift.append(0); h_fisher.append(0); h_logit.append(0); sig_true.append(0)
            continue
        # lift 순위 (기존)
        h_lift.append(1 if comm[0].tool_id == CAUSE_TOOL else 0)
        # Fisher p 오름차순 순위 (유의성 기준)
        by_p = sorted(comm, key=lambda c: c.p_value)
        h_fisher.append(1 if by_p[0].tool_id == CAUSE_TOOL else 0)
        true_cell = next((c for c in comm if c.tool_id == CAUSE_TOOL), None)
        sig_true.append(1 if (true_cell and true_cell.p_value < 0.01) else 0)
        # 로지스틱 계수 순위 (교락 통제)
        logit = commonality_logistic(results, top_k=1)
        h_logit.append(1 if (logit and logit[0][0] == CAUSE_TOOL) else 0)

    return MethodCell(
        correlation=correlation,
        n_lots=n_lots,
        top1_lift=float(np.mean(h_lift)),
        top1_fisher=float(np.mean(h_fisher)),
        top1_logit=float(np.mean(h_logit)),
        sig_rate_true=float(np.mean(sig_true)),
        defects_mean=float(np.mean(defects)),
        n_seeds=n_seeds,
    )


def run_method_comparison(
    correlations=(0.9, 0.5, 0.1),
    lot_counts=(10, 20, 40, 80),
    n_seeds: int = 24,
) -> list[MethodCell]:
    return [
        run_method_cell(c, n, n_seeds)
        for c in correlations
        for n in lot_counts
    ]


def print_method_comparison(cells: list[MethodCell], correlations, lot_counts) -> None:
    idx = {(c.correlation, c.n_lots): c for c in cells}
    print("\n" + "=" * 78)
    print("방법 비교 — top-1 성공률: lift(효과크기) vs Fisher(유의성) vs 로지스틱(교락통제)")
    print("=" * 78)
    for corr in correlations:
        print(f"\ncorrelation = {corr}")
        print(f"{'n_lots':>7} {'불량~':>6} {'lift':>7} {'Fisher':>7} {'logit':>7}   {'참장비 p<0.01':>12}")
        print("-" * 55)
        for n in lot_counts:
            c = idx[(corr, n)]
            print(f"{n:>7} {c.defects_mean:>6.0f} {c.top1_lift:>7.2f} {c.top1_fisher:>7.2f} "
                  f"{c.top1_logit:>7.2f}   {c.sig_rate_true:>12.2f}")
    # 교락 구간(작은 표본)에서의 개선 요약
    print("\n· 교락 구간 개선 (lift → logit), n_lots=10~20 평균:")
    for corr in correlations:
        small = [idx[(corr, n)] for n in lot_counts if n <= 20 and (corr, n) in idx]
        if small:
            l = np.mean([c.top1_lift for c in small])
            g = np.mean([c.top1_logit for c in small])
            print(f"    corr={corr}: {l:.2f} → {g:.2f}  ({'+' if g>=l else ''}{(g-l):.2f})")


def main() -> None:
    correlations = (0.9, 0.7, 0.5, 0.3, 0.1)
    lot_counts = (10, 20, 40, 80, 160)
    n_seeds = 24
    cells = run_grid(correlations, lot_counts, n_seeds)
    print_grid(cells, correlations, lot_counts)
    summarize(cells, lot_counts, correlations)

    # 3-방법 비교 (교락 구간 중심의 축약 격자)
    m_corr = (0.9, 0.5, 0.1)
    m_lots = (10, 20, 40, 80)
    mcells = run_method_comparison(m_corr, m_lots, n_seeds)
    print_method_comparison(mcells, m_corr, m_lots)


if __name__ == "__main__":
    main()

"""
FabScope — Fab Simulator (Phase 2)
===================================

defect_sensor_mapping의 규칙을 실제 센서 스트림으로 바꾸는 시뮬레이터.

## 세 가지 책임

1. **Lot router**: WM-811K의 각 wafer(lot)에 route(거쳐갈 장비 순서)를 배정한다.
2. **이상 주입 스케줄러**: 특정 장비에 특정 시각부터 이상을 주입한다.
   그 시간대에 그 장비를 거친 lot의 wafer가 해당 defect 라벨을 받는다.무
3. **센서 값 생성기**: 매 tick마다 모든 장비의 250채널 값을 생성한다.
   - 정상 장비: baseline N(mean, std) + 드리프트
   - 이상 장비: 핵심 센서에 deviation 적용
   - 배경 채널: 항상 baseline 거동 (오경보의 원천)

## 핵심 설계 결정

- **정답을 알고 생성한다.** 어느 장비가 언제 어떤 이상인지 우리가 정하므로,
  파이프라인의 탐지 지연·정확도를 정량 측정할 수 있다. 이것이 시뮬레이션의 고유 가치.
- **equipment_correlation로 노이즈를 준다.** defect wafer의 80%만 실제 이상 장비를
  거치고, 20%는 다른 경로로 산포 → commonality analysis가 100% 정답이 아닌
  현실적 문제를 풀도록.
- **재현 가능.** 모든 랜덤은 seed로 결정론적. 같은 입력 → 같은 스트림.

## 출력

각 tick마다 SensorReading 레코드들 (tool_id, sensor, value, timestamp, is_anomaly).
Kafka producer가 이것을 직렬화해 전송한다 (Phase 2 다음 단계).

## 상세 근거

Notion 16번(방법론), 19번(설계 결정), 22번(센서 상세) 참조.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from defect_sensor_mapping import (
    DEFECT_RULES,
    MISSING_RATE,
    NORMAL_LABEL,
    SAMPLING_RATE_BASELINE,
    TOOL_UTILIZATION,
    Deviation,
    SensorSpec,
    Tool,
    build_fab,
)

# ==============================================================================
# 1. Route 배정
# ==============================================================================
#
# wafer는 lot(25장) 단위로 이동하며 각 공정 모듈에서 장비 1대에 배정된다.
# 실제 fab의 공정 순서를 단순화한 표준 route를 쓴다.

# 표준 공정 순서 (실제로는 수백 단계지만, 우리는 6개 모듈 각 1회로 단순화)
STANDARD_FLOW = [
    "Photolithography",
    "Etch",
    "ThinFilm",
    "Diffusion",
    "CMP",
    "Cleaning",
]


@dataclass(frozen=True)
class Route:
    """한 lot이 거쳐가는 장비의 순서."""

    lot_id: str
    steps: dict[str, str]  # module -> tool_id  예: {"Etch": "ETCH-07"}

    def uses_tool(self, tool_id: str) -> bool:
        return tool_id in self.steps.values()

    def tool_for(self, module: str) -> str:
        return self.steps[module]


class LotRouter:
    """lot마다 route를 배정한다.

    각 모듈에서 어느 장비를 쓸지는 랜덤이되, 장비별 부하가 비슷하도록 순환 배정에
    약간의 무작위를 섞는다 (실제 fab의 dispatching을 단순 모사).
    """

    def __init__(self, tools: dict[str, Tool], seed: int = 42):
        self.tools = tools
        self.rng = np.random.default_rng(seed)

        # 모듈별 장비 목록
        self.tools_by_module: dict[str, list[str]] = {}
        for tid, tool in tools.items():
            self.tools_by_module.setdefault(tool.module, []).append(tid)
        for module in self.tools_by_module:
            self.tools_by_module[module].sort()

    def assign(self, lot_id: str) -> Route:
        """한 lot에 route를 배정한다."""
        steps: dict[str, str] = {}
        for module in STANDARD_FLOW:
            candidates = self.tools_by_module[module]
            chosen = candidates[self.rng.integers(len(candidates))]
            steps[module] = chosen
        return Route(lot_id=lot_id, steps=steps)


# ==============================================================================
# 2. 이상 주입 스케줄
# ==============================================================================


@dataclass
class AnomalyInjection:
    """특정 장비에 특정 시각부터 주입되는 이상.

    이것이 '정답지'다. 이 장비를 이 시간대에 거친 lot의 wafer가 해당 defect가 된다.
    """

    tool_id: str
    defect: str  # WM-811K defect 라벨
    start_tick: int
    end_tick: int  # 이 tick까지 지속
    ramp_ticks: int = 0  # 이상이 서서히 심해지는 구간 (0이면 즉시)

    def is_active(self, tick: int) -> bool:
        return self.start_tick <= tick <= self.end_tick

    def severity(self, tick: int) -> float:
        """현재 이상의 강도 (0~1). ramp 구간에서는 선형 증가.

        급성 고장은 즉시 1.0, 점진적 열화는 서서히 올라간다.
        """
        if not self.is_active(tick):
            return 0.0
        if self.ramp_ticks <= 0:
            return 1.0
        elapsed = tick - self.start_tick
        return min(1.0, elapsed / self.ramp_ticks)


class AnomalyScheduler:
    """이상 주입 스케줄을 관리한다.

    한 장비에 동시에 하나의 이상만 활성화된다고 가정 (단순화).
    """

    def __init__(self):
        self.injections: list[AnomalyInjection] = []

    def add(self, injection: AnomalyInjection) -> None:
        self.injections.append(injection)

    def active_for(self, tool_id: str, tick: int) -> AnomalyInjection | None:
        """해당 장비에 지금 활성화된 이상을 반환 (없으면 None)."""
        for inj in self.injections:
            if inj.tool_id == tool_id and inj.is_active(tick):
                return inj
        return None

    def active_tools(self, tick: int) -> dict[str, AnomalyInjection]:
        """현재 tick에 이상이 활성화된 모든 장비."""
        return {
            inj.tool_id: inj
            for inj in self.injections
            if inj.is_active(tick)
        }


# ==============================================================================
# 3. 센서 값 생성
# ==============================================================================


class SensorValueGenerator:
    """장비 하나의 센서 값을 생성한다.

    정상: N(baseline, sigma) + 드리프트
    이상: 핵심 센서에 deviation 적용 (severity로 스케일)
    배경: 항상 정상 거동 (deviation 없음)
    """

    def __init__(self, tool: Tool, seed: int):
        self.tool = tool
        self.rng = np.random.default_rng(seed)

        # 드리프트 상태 — 채널별 누적 드리프트 (랜덤워크)
        self._drift: dict[str, float] = {name: 0.0 for name in tool.all_sensors}

        # 이상별 deviation 인덱스 (핵심 센서만)
        self._deviation_map: dict[str, dict[str, Deviation]] = {}
        for defect, rule in DEFECT_RULES.items():
            if rule.module != tool.module:
                continue
            self._deviation_map[defect] = {d.sensor: d for d in rule.deviations}

    def _update_drift(self, spec: SensorSpec, dt_days: float) -> None:
        """드리프트를 랜덤워크로 누적. PM 리셋은 별도 처리."""
        if spec.drift_per_day <= 0:
            return
        # 하루당 drift_per_day 비율만큼 표준편차를 갖는 랜덤워크
        step = self.rng.normal(0, spec.drift_per_day * spec.baseline * dt_days)
        self._drift[spec.name] += step

    def reset_drift(self, sensor_name: str) -> None:
        """PM/세정 시 드리프트 리셋."""
        self._drift[sensor_name] = 0.0

    def generate(
        self,
        tick: int,
        injection: AnomalyInjection | None,
        dt_days: float = 1.0 / 86400,  # 1 tick = 1초 (1 Hz 기준)
    ) -> dict[str, float | None]:
        """이 tick의 모든 채널 값을 생성한다.

        반환: {sensor_name: value}. 결측이면 value=None.
        """
        values: dict[str, float | None] = {}
        severity = injection.severity(tick) if injection else 0.0
        active_devs = (
            self._deviation_map.get(injection.defect, {}) if injection else {}
        )

        for name, spec in self.tool.all_sensors.items():
            # 결측 처리 (SENSOR 고장/누락)
            if self.rng.random() < MISSING_RATE:
                values[name] = None
                continue

            # 카운터는 노이즈 없이 처리 (pad_life_count 등)
            if spec.sigma == 0:
                values[name] = spec.baseline
                continue

            # 드리프트 갱신
            self._update_drift(spec, dt_days)
            drift = self._drift[name]

            # 이상이 이 센서에 적용되는가
            dev = active_devs.get(name)
            if dev is not None and severity > 0:
                # deviation을 severity로 스케일 (ramp 반영)
                mean = spec.baseline + dev.mean_shift_sigma * spec.sigma * severity
                std = spec.sigma * (1.0 + (dev.std_multiplier - 1.0) * severity)
            else:
                mean = spec.baseline
                std = spec.sigma

            value = self.rng.normal(mean + drift, std)
            values[name] = float(value)

        return values


# ==============================================================================
# 4. 센서 판독 레코드
# ==============================================================================


@dataclass(frozen=True)
class SensorReading:
    """스트림으로 나가는 단위 레코드 (Kafka 메시지 1건에 대응)."""

    timestamp: float  # epoch seconds
    tick: int
    tool_id: str
    module: str
    sensor: str
    value: float
    # 아래는 정답지 (실제 배포에선 없지만, 평가용으로 포함)
    is_anomaly_tool: bool  # 이 장비에 이상이 주입됐는가
    injected_defect: str | None  # 주입된 defect (없으면 None)

    def to_dict(self) -> dict:
        """Kafka 전송용 직렬화."""
        return {
            "ts": self.timestamp,
            "tick": self.tick,
            "tool": self.tool_id,
            "module": self.module,
            "sensor": self.sensor,
            "value": round(self.value, 4),
            # ground truth (별도 토픽이나 메타로 분리 가능)
            "_gt_anomaly": self.is_anomaly_tool,
            "_gt_defect": self.injected_defect,
        }


# ==============================================================================
# 5. Wafer 결과 생성 (equipment_correlation 반영)
# ==============================================================================
#
# ⭐ 여기가 시뮬레이터의 핵심 트릭이다.
#
# 이상 장비(ETCH-07)를 거친 lot의 wafer가 전부 defect가 되는 게 아니다.
# equipment_correlation = 0.8이면:
#   - defect wafer의 80%는 실제로 그 이상 장비를 거쳤고 (진짜 인과)
#   - 20%는 다른 경로로 생긴 산발적 defect (노이즈)
#
# 이 노이즈가 commonality analysis를 현실적 문제로 만든다. 100% 깨끗한 신호면
# 원인 장비 찾기가 너무 쉬워서 파이프라인 검증의 의미가 없다.


@dataclass(frozen=True)
class WaferResult:
    """한 wafer의 검사 결과 (wafer_results 테이블에 대응)."""

    wafer_id: str
    lot_id: str
    tick: int  # 이 wafer가 처리된 시각
    label: str  # WM-811K defect 라벨 (또는 'none')
    route: dict[str, str]  # 이 wafer가 거친 장비들
    caused_by_tool: str | None  # 실제 원인 장비 (정답지)


class WaferResultGenerator:
    """lot의 route와 이상 스케줄로부터 wafer 결과를 생성한다."""

    def __init__(self, scheduler: AnomalyScheduler, seed: int = 42):
        self.scheduler = scheduler
        self.rng = np.random.default_rng(seed)

    def generate_for_lot(
        self, route: Route, process_tick: int, wafers_per_lot: int = 25
    ) -> list[WaferResult]:
        """한 lot의 wafer들 결과를 생성한다.

        process_tick: 이 lot이 (문제의) 공정을 거친 시각.
        실제로는 각 모듈마다 시각이 다르지만, 단순화해 하나의 tick으로 본다.
        """
        results: list[WaferResult] = []

        # 이 lot이 거친 장비 중 이상이 활성화된 것이 있는가
        active_defect: str | None = None
        active_tool: str | None = None
        for module, tool_id in route.steps.items():
            inj = self.scheduler.active_for(tool_id, process_tick)
            if inj is not None:
                active_defect = inj.defect
                active_tool = tool_id
                break  # 첫 번째 이상 장비 (단순화: lot당 이상 1개)

        for w in range(wafers_per_lot):
            wafer_id = f"{route.lot_id}-W{w:02d}"

            if active_defect is not None:
                # 이상 장비를 거친 lot — equipment_correlation 확률로 defect 발현
                corr = DEFECT_RULES[active_defect].equipment_correlation
                if self.rng.random() < corr:
                    # 진짜 인과: 이 장비가 원인
                    results.append(
                        WaferResult(
                            wafer_id=wafer_id,
                            lot_id=route.lot_id,
                            tick=process_tick,
                            label=active_defect,
                            route=dict(route.steps),
                            caused_by_tool=active_tool,
                        )
                    )
                else:
                    # 장비를 거쳤지만 이번엔 정상 (correlation < 1의 효과)
                    results.append(
                        WaferResult(
                            wafer_id=wafer_id,
                            lot_id=route.lot_id,
                            tick=process_tick,
                            label=NORMAL_LABEL,
                            route=dict(route.steps),
                            caused_by_tool=None,
                        )
                    )
            else:
                # 이상 장비를 안 거친 lot — 대부분 정상, 극소수 산발적 defect (노이즈)
                if self.rng.random() < BACKGROUND_DEFECT_RATE:
                    # 원인 불명의 산발적 defect (commonality analysis의 노이즈)
                    spurious = self._random_defect()
                    results.append(
                        WaferResult(
                            wafer_id=wafer_id,
                            lot_id=route.lot_id,
                            tick=process_tick,
                            label=spurious,
                            route=dict(route.steps),
                            caused_by_tool=None,  # 특정 장비가 원인 아님
                        )
                    )
                else:
                    results.append(
                        WaferResult(
                            wafer_id=wafer_id,
                            lot_id=route.lot_id,
                            tick=process_tick,
                            label=NORMAL_LABEL,
                            route=dict(route.steps),
                            caused_by_tool=None,
                        )
                    )

        return results

    def _random_defect(self) -> str:
        """산발적(원인 불명) defect 하나를 랜덤 선택."""
        defects = list(DEFECT_RULES.keys())
        return defects[self.rng.integers(len(defects))]


# 이상 장비를 안 거친 lot에서 산발적으로 defect가 나올 확률.
# 실제 fab의 baseline 불량률을 모사 (매우 낮음).
BACKGROUND_DEFECT_RATE = 0.01


# ==============================================================================
# 6. 메인 시뮬레이터 엔진
# ==============================================================================


class FabSimulator:
    """전체를 묶는 엔진.

    tick마다:
      1. 모든 장비의 250채널 값 생성 → SensorReading 스트림
      2. (lot이 공정을 거치는 시점에) wafer 결과 생성

    사용:
        sim = FabSimulator(seed=42)
        sim.schedule_anomaly("ETCH-07", "Edge-Ring", start_tick=100, end_tick=500)
        for reading in sim.run(n_ticks=1000):
            producer.send(reading.to_dict())
    """

    def __init__(
        self,
        seed: int = 42,
        sampling_hz: float = SAMPLING_RATE_BASELINE,
        utilization: float = TOOL_UTILIZATION,
        start_time: float | None = None,
    ):
        self.seed = seed
        self.sampling_hz = sampling_hz
        self.utilization = utilization
        self.start_time = start_time if start_time is not None else time.time()
        self.dt = 1.0 / sampling_hz  # tick 간 실제 시간(초)

        # 구성
        self.tools = build_fab()
        self.router = LotRouter(self.tools, seed=seed)
        self.scheduler = AnomalyScheduler()
        self.wafer_gen = WaferResultGenerator(self.scheduler, seed=seed + 1)

        # 장비별 값 생성기 (각각 다른 시드)
        self.generators: dict[str, SensorValueGenerator] = {}
        for i, (tid, tool) in enumerate(self.tools.items()):
            self.generators[tid] = SensorValueGenerator(tool, seed=seed + 100 + i)

        # 가동 상태 — 각 tick에 어떤 장비가 가동 중인지 (utilization 반영)
        self._util_rng = np.random.default_rng(seed + 2)

        # lot 카운터
        self._lot_counter = 0

    def schedule_anomaly(
        self,
        tool_id: str,
        defect: str,
        start_tick: int,
        end_tick: int,
        ramp_ticks: int = 0,
    ) -> None:
        """이상 주입을 예약한다."""
        if tool_id not in self.tools:
            raise ValueError(f"알 수 없는 장비: {tool_id}")
        if defect not in DEFECT_RULES:
            raise ValueError(f"알 수 없는 defect: {defect}")
        # 장비 모듈과 defect 모듈이 맞는지 확인
        expected_module = DEFECT_RULES[defect].module
        actual_module = self.tools[tool_id].module
        if expected_module != actual_module:
            raise ValueError(
                f"{defect}는 {expected_module} 모듈인데 {tool_id}는 {actual_module}"
            )
        self.scheduler.add(
            AnomalyInjection(
                tool_id=tool_id,
                defect=defect,
                start_tick=start_tick,
                end_tick=end_tick,
                ramp_ticks=ramp_ticks,
            )
        )

    def _timestamp(self, tick: int) -> float:
        return self.start_time + tick * self.dt

    def _is_running(self, tool_id: str, tick: int) -> bool:
        """이 장비가 이 tick에 가동 중인가 (utilization 확률).

        가동률 0.8이면 평균적으로 80%의 tick에서 신호를 낸다.
        이상이 주입된 장비는 항상 가동 (이상을 놓치지 않기 위해).
        """
        if self.scheduler.active_for(tool_id, tick) is not None:
            return True
        return self._util_rng.random() < self.utilization

    def tick_readings(self, tick: int) -> list[SensorReading]:
        """한 tick의 모든 센서 판독을 생성한다."""
        readings: list[SensorReading] = []
        ts = self._timestamp(tick)

        for tid, tool in self.tools.items():
            if not self._is_running(tid, tick):
                continue

            injection = self.scheduler.active_for(tid, tick)
            gen = self.generators[tid]
            values = gen.generate(tick, injection, dt_days=self.dt / 86400)

            for sensor, value in values.items():
                if value is None:  # 결측
                    continue
                readings.append(
                    SensorReading(
                        timestamp=ts,
                        tick=tick,
                        tool_id=tid,
                        module=tool.module,
                        sensor=sensor,
                        value=value,
                        is_anomaly_tool=injection is not None,
                        injected_defect=injection.defect if injection else None,
                    )
                )

        return readings

    def run(self, n_ticks: int):
        """제너레이터로 센서 스트림을 흘린다.

        yield: SensorReading (하나씩)
        """
        for tick in range(n_ticks):
            yield from self.tick_readings(tick)

    def process_lots(self, n_lots: int, at_tick: int) -> list[WaferResult]:
        """n_lots개의 lot을 at_tick 시점에 공정 처리하고 wafer 결과를 생성한다.

        실제로는 lot이 시간에 걸쳐 흐르지만, 여기선 배치로 단순화.
        WM-811K 데이터와 연결할 때는 각 wafer map을 이 결과에 매핑한다.
        """
        results: list[WaferResult] = []
        for _ in range(n_lots):
            lot_id = f"LOT-{self._lot_counter:05d}"
            self._lot_counter += 1
            route = self.router.assign(lot_id)
            results.extend(self.wafer_gen.generate_for_lot(route, at_tick))
        return results


# ==============================================================================
# 7. Commonality Analysis (원인 장비 지목)
# ==============================================================================
#
# 이것이 파이프라인의 최종 산출물이자, 시뮬레이터가 제대로 작동하는지의 검증 지표다.
#
# 원리: 불량 wafer들의 route를 모아 "공통으로 거친 장비"를 찾는다.
# 정상 wafer 대비 불량 wafer에서 특정 장비의 출현 빈도가 유의하게 높으면 그 장비가 용의자.


@dataclass
class CommonalityResult:
    """장비별 용의도 점수."""

    tool_id: str
    defect_pass_count: int  # 불량 wafer 중 이 장비를 거친 수
    total_pass_count: int  # 전체 wafer 중 이 장비를 거친 수
    defect_rate: float  # 이 장비를 거친 wafer의 불량률
    lift: float  # (이 장비 불량률) / (전체 불량률) — 높을수록 용의


def commonality_analysis(
    wafer_results: list[WaferResult],
    top_k: int = 5,
) -> list[CommonalityResult]:
    """불량 wafer들의 공통 장비를 찾는다.

    lift = P(defect | 이 장비 경유) / P(defect) 로 용의도를 계산.
    lift가 1보다 크게 높으면 그 장비가 불량과 연관.
    """
    from collections import defaultdict

    tool_total: dict[str, int] = defaultdict(int)
    tool_defect: dict[str, int] = defaultdict(int)

    n_total = len(wafer_results)
    n_defect = sum(1 for r in wafer_results if r.label != NORMAL_LABEL)
    if n_total == 0 or n_defect == 0:
        return []

    base_defect_rate = n_defect / n_total

    for r in wafer_results:
        is_defect = r.label != NORMAL_LABEL
        for tool_id in r.route.values():
            tool_total[tool_id] += 1
            if is_defect:
                tool_defect[tool_id] += 1

    results: list[CommonalityResult] = []
    for tool_id, total in tool_total.items():
        d = tool_defect[tool_id]
        rate = d / total if total else 0.0
        lift = rate / base_defect_rate if base_defect_rate else 0.0
        results.append(
            CommonalityResult(
                tool_id=tool_id,
                defect_pass_count=d,
                total_pass_count=total,
                defect_rate=rate,
                lift=lift,
            )
        )

    results.sort(key=lambda x: x.lift, reverse=True)
    return results[:top_k]


# ==============================================================================
# 8. Self-test
# ==============================================================================


def _selftest():
    print("=" * 78)
    print("FabSimulator Self-Test")
    print("=" * 78)

    # --- 구성 ---
    sim = FabSimulator(seed=42)
    print(f"\n[구성] 장비 {len(sim.tools)}대")
    total_channels = sum(t.channel_count for t in sim.tools.values())
    print(f"       총 채널 {total_channels:,}개")

    # --- Route 배정 ---
    print("\n[Route 배정] 샘플 3개")
    for i in range(3):
        route = sim.router.assign(f"LOT-TEST-{i}")
        path = " → ".join(route.steps[m] for m in STANDARD_FLOW)
        print(f"  {route.lot_id}: {path}")

    # --- 이상 주입 ---
    print("\n[이상 주입] ETCH-07에 Edge-Ring (tick 100~500, ramp 50)")
    sim.schedule_anomaly("ETCH-07", "Edge-Ring", start_tick=100, end_tick=500, ramp_ticks=50)

    # --- 센서 스트림 (한 tick) ---
    print("\n[센서 스트림] tick 200의 판독 수")
    readings_200 = sim.tick_readings(200)
    print(f"  총 {len(readings_200):,}건 (가동률 {sim.utilization} 반영)")

    # ETCH-07의 이상 센서 확인
    etch07 = [r for r in readings_200 if r.tool_id == "ETCH-07"]
    anomaly_readings = [r for r in etch07 if r.injected_defect]
    print(f"  ETCH-07: {len(etch07)}채널, is_anomaly={anomaly_readings[0].is_anomaly_tool if anomaly_readings else 'N/A'}")

    # chamber_pressure가 실제로 상승했는지 (baseline 40, Edge-Ring +2.4σ)
    cp_normal = []
    cp_anomaly = []
    for tick in range(50, 100):  # 이상 전
        for r in sim.tick_readings(tick):
            if r.tool_id == "ETCH-07" and r.sensor == "chamber_pressure":
                cp_normal.append(r.value)
    for tick in range(200, 250):  # 이상 후 (ramp 완료)
        for r in sim.tick_readings(tick):
            if r.tool_id == "ETCH-07" and r.sensor == "chamber_pressure":
                cp_anomaly.append(r.value)
    if cp_normal and cp_anomaly:
        print(f"\n  chamber_pressure 정상 평균: {np.mean(cp_normal):.2f} mTorr (baseline 40)")
        print(f"  chamber_pressure 이상 평균: {np.mean(cp_anomaly):.2f} mTorr (기대 +2.4σ ≈ 43)")
        shift = np.mean(cp_anomaly) - np.mean(cp_normal)
        print(f"  → 이동량: {shift:+.2f} mTorr (기대 ~3.0)")

    # --- Wafer 결과 + Commonality ---
    print("\n[Wafer 결과] tick 200에서 lot 200개 처리")
    # 이상이 확실히 걸리도록 ETCH-07을 지나는 lot을 충분히 생성
    results = sim.process_lots(n_lots=200, at_tick=200)
    n_defect = sum(1 for r in results if r.label != NORMAL_LABEL)
    print(f"  총 wafer {len(results):,}장, 불량 {n_defect}장 ({n_defect/len(results):.1%})")

    # 불량 라벨 분포
    from collections import Counter
    label_dist = Counter(r.label for r in results if r.label != NORMAL_LABEL)
    print(f"  불량 분포: {dict(label_dist)}")

    print("\n[Commonality Analysis] 원인 장비 지목")
    comm = commonality_analysis(results, top_k=5)
    print(f"  {'장비':<12} {'lift':<8} {'불량률':<10} {'경유(불량/전체)'}")
    print("  " + "-" * 50)
    for c in comm:
        marker = " ⭐ 정답" if c.tool_id == "ETCH-07" else ""
        print(
            f"  {c.tool_id:<12} {c.lift:<8.2f} {c.defect_rate:<10.1%} "
            f"{c.defect_pass_count}/{c.total_pass_count}{marker}"
        )

    top_suspect = comm[0].tool_id if comm else None
    if top_suspect == "ETCH-07":
        print("\n  ✅ 원인 장비(ETCH-07)를 top-1으로 정확히 지목")
    else:
        print(f"\n  ⚠️ top-1이 {top_suspect} (정답 ETCH-07)")

    # --- 생성량/오경보 ---
    print("\n[생성량 검증]")
    from defect_sensor_mapping import expected_generation_rate, expected_false_alarm_rate
    gen_rate = expected_generation_rate(sampling_hz=1.0)
    far = expected_false_alarm_rate(sampling_hz=1.0)
    print(f"  기준 생성량: {gen_rate:,.0f} msg/s")
    print(f"  실측(tick 200): {len(readings_200):,}건/tick → {len(readings_200)*1.0:,.0f} msg/s")
    print(f"  1차 3σ 오경보 기준선: {far:.1f} 건/초")

    # --- 결정론 확인 ---
    print("\n[결정론 검증]")
    sim2 = FabSimulator(seed=42)
    sim2.schedule_anomaly("ETCH-07", "Edge-Ring", start_tick=100, end_tick=500, ramp_ticks=50)
    r1 = sim.tick_readings(300)
    r2 = sim2.tick_readings(300)
    vals1 = sorted((r.tool_id, r.sensor, round(r.value, 6)) for r in r1)
    vals2 = sorted((r.tool_id, r.sensor, round(r.value, 6)) for r in r2)
    # 주의: sim은 이미 여러 tick을 돌려 RNG 상태가 다름. 새 인스턴스로 같은 tick 비교
    sim3 = FabSimulator(seed=42)
    sim3.schedule_anomaly("ETCH-07", "Edge-Ring", start_tick=100, end_tick=500, ramp_ticks=50)
    r3 = sim3.tick_readings(300)
    sim4 = FabSimulator(seed=42)
    sim4.schedule_anomaly("ETCH-07", "Edge-Ring", start_tick=100, end_tick=500, ramp_ticks=50)
    r4 = sim4.tick_readings(300)
    v3 = sorted((r.tool_id, r.sensor, round(r.value, 6)) for r in r3)
    v4 = sorted((r.tool_id, r.sensor, round(r.value, 6)) for r in r4)
    print(f"  같은 seed, 같은 tick 재생성 일치: {v3 == v4}")

    print("\n" + "=" * 78)
    print("Self-test 완료")
    print("=" * 78)


if __name__ == "__main__":
    _selftest()

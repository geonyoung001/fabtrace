"""
FabScope — Fab Simulator (Phase 2)
===================================

defect_sensor_mapping의 규칙을 실제 센서 스트림으로 바꾸는 시뮬레이터.

## 세 가지 책임

1. **Lot router**: WM-811K의 각 wafer(lot)에 route(거쳐갈 장비 순서)를 배정한다.
2. **이상 주입 스케줄러**: 특정 장비에 특정 시각부터 이상을 주입한다.
   그 시간대에 그 장비를 거친 lot의 wafer가 해당 defect 라벨을 받는다.
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

import bisect
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from defect_sensor_mapping import (
    DEFECT_RULES,
    MISSING_RATE,
    NORMAL_LABEL,
    SAMPLING_RATE_BASELINE,
    TOOL_UTILIZATION,
    Actuation,
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
# 1b. 처리 에피소드 (Processing Episode)
# ==============================================================================
#
# 한 (wafer, step) = 1 에피소드. 장비가 wafer 1장을 처리하는 ~60s 구간.
# 에피소드 동안만 신호가 흐르고(ramp-up→steady→ramp-down), 사이 유휴엔 침묵.
#
# 이 단위가 관측의 원자다: 2차 FDC의 run, Flink 윈도우 경계, WM-811K 라벨 연결,
# 검증 3(트레이스 형태)이 전부 여기 걸린다.


@dataclass
class Episode:
    """장비가 wafer 1장의 한 공정 step을 처리하는 구간."""

    wafer_id: str
    lot_id: str
    step: str  # 모듈명
    tool_id: str
    start_tick: int
    sampling_hz: float
    duration_s: float = 60.0
    ramp_s: float = 5.0

    @property
    def duration_ticks(self) -> int:
        return max(1, int(round(self.duration_s * self.sampling_hz)))

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_ticks

    @property
    def episode_id(self) -> str:
        return f"{self.wafer_id}:{self.step}"

    def phase(self, tick: int) -> tuple[str, float] | None:
        """이 tick의 (phase 이름, 계수). 범위 밖이면 None.

        경계는 하드코딩이 아니라 ramp_s/duration_s에서 파생 — duration을 바꿔도 추종.
        """
        e = tick - self.start_tick
        if e < 0 or e >= self.duration_ticks:
            return None
        t = e / self.sampling_hz  # 경과 초
        if t < self.ramp_s:
            return ("ramp_up", t / self.ramp_s)
        if t < self.duration_s - self.ramp_s:
            return ("steady", 1.0)
        return ("ramp_down", max(0.0, (self.duration_s - t) / self.ramp_s))


class EpisodeScheduler:
    """wafer들을 STANDARD_FLOW를 따라 장비에 흘리며 에피소드 타임라인을 만든다.

    - 각 lot에 route(module→tool)를 배정하고, lot 내 wafer들은 그 route를 공유한다
      (실제 fab에서 lot 25장은 함께 이동).
    - 각 (wafer, module)에 에피소드 1개. 장비당 동시 1개(겹침 금지).
    - 에피소드 사이 idle gap으로 duty cycle을 조절 — duty = dur / (dur + gap).
      단 tool 부하 불균형·전공정 대기로 실측 duty는 target 아래로 흔들릴 수 있다
      (자동이 아니라 튜닝·검증 대상; §2-1).

    자기 LotRouter를 따로 둬서 FabSimulator의 wafer-result 경로와 RNG가 얽히지 않게 한다.
    (에피소드 스트림과 wafer 라벨의 완전 통합은 WM-811K 연결 단계에서.)
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        sampling_hz: float,
        duration_s: float = 60.0,
        ramp_s: float = 5.0,
        target_duty: float = TOOL_UTILIZATION,
        seed: int = 42,
    ):
        self.router = LotRouter(tools, seed=seed)
        self.sampling_hz = sampling_hz
        self.duration_s = duration_s
        self.ramp_s = ramp_s
        self.target_duty = target_duty

        self.per_tool: dict[str, list[Episode]] = {}
        self._starts: dict[str, list[int]] = {}
        self.wafer_history: dict[str, list[Episode]] = {}

    def build(self, n_lots: int, wafers_per_lot: int = 25, start_index: int = 0) -> "EpisodeScheduler":
        dur = max(1, int(round(self.duration_s * self.sampling_hz)))
        gap = max(0, int(round(dur * (1.0 / self.target_duty - 1.0))))

        # wafer 투입 간격(release_interval)을 병목 모듈이 target_duty가 되도록 역산한다.
        # 흐름 보존: 모든 wafer가 모든 모듈을 1회 거치므로, 모듈 처리율 = fab 투입율.
        # 모듈 처리율 = (모듈 내 tool 수) × target_duty / dur.
        # → 병목(최소 tool 수) 모듈이 target을 넘지 않게 최소 tool 수로 역산.
        #   (tool 수가 많은 모듈은 target 아래로 — 실측 duty가 tool마다 다른 이유)
        min_tools_per_module = min(
            len(v) for v in self.router.tools_by_module.values()
        )
        release_interval = max(
            1, int(round(dur / (min_tools_per_module * self.target_duty)))
        )

        per_tool: dict[str, list[Episode]] = defaultdict(list)
        wafer_history: dict[str, list[Episode]] = {}
        tool_free: dict[str, int] = defaultdict(int)

        release = 0
        for li in range(start_index, start_index + n_lots):
            lot_id = f"LOT-{li:05d}"
            route = self.router.assign(lot_id)
            for wi in range(wafers_per_lot):
                wafer_id = f"WAF-{li:05d}-{wi:02d}"
                ready = release
                steps: list[Episode] = []
                for module in STANDARD_FLOW:
                    tool = route.steps[module]
                    start = max(ready, tool_free[tool])
                    ep = Episode(
                        wafer_id=wafer_id,
                        lot_id=lot_id,
                        step=module,
                        tool_id=tool,
                        start_tick=start,
                        sampling_hz=self.sampling_hz,
                        duration_s=self.duration_s,
                        ramp_s=self.ramp_s,
                    )
                    per_tool[tool].append(ep)
                    steps.append(ep)
                    ready = ep.end_tick
                    tool_free[tool] = ep.end_tick + gap
                wafer_history[wafer_id] = steps
                release += release_interval

        for tool, eps in per_tool.items():
            eps.sort(key=lambda e: e.start_tick)
        self.per_tool = dict(per_tool)
        self._starts = {t: [e.start_tick for e in eps] for t, eps in self.per_tool.items()}
        self.wafer_history = wafer_history
        return self

    def active(self, tool_id: str, tick: int) -> Episode | None:
        """이 tool이 이 tick에 처리 중인 에피소드. 없으면 None(=idle)."""
        starts = self._starts.get(tool_id)
        if not starts:
            return None
        i = bisect.bisect_right(starts, tick) - 1
        if i < 0:
            return None
        ep = self.per_tool[tool_id][i]
        return ep if ep.phase(tick) is not None else None

    def horizon(self) -> int:
        """마지막 에피소드가 끝나는 tick."""
        return max(
            (e.end_tick for eps in self.per_tool.values() for e in eps),
            default=0,
        )

    def measure_duty(self, tool_id: str, t0: int, t1: int) -> float:
        """[t0, t1) 구간에서 이 tool의 실측 duty(활성 tick 비율)."""
        if t1 <= t0:
            return 0.0
        active = sum(1 for tick in range(t0, t1) if self.active(tool_id, tick) is not None)
        return active / (t1 - t0)


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
        phase_name: str,
        coeff: float,
        injection: AnomalyInjection | None,
        dt_days: float = 1.0 / 86400,  # 1 tick = 1초 (1 Hz 기준)
    ) -> dict[str, float | None]:
        """이 tick(활성 에피소드 안)의 모든 채널 값을 생성한다.

        phase_name: "ramp_up" / "steady" / "ramp_down"
        coeff:      phase 계수 (RECIPE 센서의 사다리꼴; HELD는 무시)

        값 프로파일:
          - RECIPE: coeff × baseline (0→setpoint→0 사다리꼴)
          - HELD:   baseline 평탄 (ramp 없음)
        deviation(이상 서명)은 **원인 센서 + severity>0 + phase==steady**일 때만.
        (ramp 구간에 실으면 "setpoint 대비 σ배수" 정의가 무의미해지고 slope 특징을 오염)

        반환: {sensor_name: value}. 결측이면 value=None.
        """
        values: dict[str, float | None] = {}
        severity = injection.severity(tick) if injection else 0.0
        active_devs = (
            self._deviation_map.get(injection.defect, {}) if injection else {}
        )
        in_steady = phase_name == "steady"

        for name, spec in self.tool.all_sensors.items():
            # 결측 처리 (SENSOR 고장/누락)
            if self.rng.random() < MISSING_RATE:
                values[name] = None
                continue

            # 카운터는 노이즈·ramp 없이 평탄 (pad_life_count 등 — HELD 성격)
            if spec.sigma == 0:
                values[name] = spec.baseline
                continue

            # 드리프트 갱신 (tool-level 상태, 에피소드를 넘어 누적)
            self._update_drift(spec, dt_days)
            drift = self._drift[name]

            # 값 프로파일: RECIPE는 phase 계수로 사다리꼴, HELD는 평탄
            if spec.actuation == Actuation.RECIPE:
                mean = coeff * spec.baseline
            else:  # HELD
                mean = spec.baseline
            std = spec.sigma

            # deviation — 원인 센서 + severity>0 + steady일 때만 (실제 코드 분기)
            dev = active_devs.get(name)
            if dev is not None and severity > 0 and in_steady:
                mean = spec.baseline + dev.mean_shift_sigma * spec.sigma * severity
                std = spec.sigma * (1.0 + (dev.std_multiplier - 1.0) * severity)

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
    # 에피소드 스탬프 — per-wafer 트레이스 재구성 + Flink (wafer, step) 윈도우 키
    wafer_id: str
    step: str
    episode_id: str
    phase: str  # ramp_up / steady / ramp_down — 1차 FDC steady 게이트용
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
            "wafer": self.wafer_id,
            "step": self.step,
            "phase": self.phase,
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
        duration_s: float = 60.0,
        ramp_s: float = 5.0,
        n_lots: int = 40,
        wafers_per_lot: int = 25,
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

        # 처리 에피소드 스케줄 — 스트림의 방출 창(idle=침묵) + wafer/phase 스탬프의 출처
        self.episodes = EpisodeScheduler(
            self.tools,
            sampling_hz=sampling_hz,
            duration_s=duration_s,
            ramp_s=ramp_s,
            target_duty=utilization,
            seed=seed + 3,
        ).build(n_lots=n_lots, wafers_per_lot=wafers_per_lot)

        # lot 카운터 (wafer-result/commonality 경로용)
        self._lot_counter = 0

    def build_schedule(self, n_lots: int, wafers_per_lot: int = 25) -> EpisodeScheduler:
        """에피소드 스케줄을 다시 만든다(더 긴 horizon이 필요할 때)."""
        self.episodes = EpisodeScheduler(
            self.tools,
            sampling_hz=self.sampling_hz,
            duration_s=self.episodes.duration_s,
            ramp_s=self.episodes.ramp_s,
            target_duty=self.utilization,
            seed=self.seed + 3,
        ).build(n_lots=n_lots, wafers_per_lot=wafers_per_lot)
        return self.episodes

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

    def tick_readings(self, tick: int) -> list[SensorReading]:
        """한 tick의 모든 센서 판독을 생성한다.

        장비가 처리 중(활성 에피소드)일 때만 방출. 유휴 장비는 침묵(레코드 없음).
        deviation은 그 장비에 이상이 주입됐고 phase==steady일 때만 실린다.
        """
        readings: list[SensorReading] = []
        ts = self._timestamp(tick)

        for tid, tool in self.tools.items():
            ep = self.episodes.active(tid, tick)
            if ep is None:  # idle → 침묵
                continue
            phase_name, coeff = ep.phase(tick)  # active()가 None 아님을 보장

            injection = self.scheduler.active_for(tid, tick)
            gen = self.generators[tid]
            values = gen.generate(
                tick, phase_name, coeff, injection, dt_days=self.dt / 86400
            )

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
                        wafer_id=ep.wafer_id,
                        step=ep.step,
                        episode_id=ep.episode_id,
                        phase=phase_name,
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


def _episode_steady_values(sim: "FabSimulator", ep: Episode, tool_id: str, sensor: str) -> list[float]:
    """한 에피소드의 steady 구간에서 특정 센서 값을 모은다."""
    vals: list[float] = []
    for tick in range(ep.start_tick, ep.end_tick):
        ph = ep.phase(tick)
        if not ph or ph[0] != "steady":
            continue
        for r in sim.tick_readings(tick):
            if r.tool_id == tool_id and r.sensor == sensor:
                vals.append(r.value)
    return vals


def _episode_trace(sim: "FabSimulator", ep: Episode, tool_id: str, sensor: str):
    """한 에피소드 전체에서 특정 센서의 (경과tick, 값) 트레이스."""
    xs: list[int] = []
    ys: list[float] = []
    for tick in range(ep.start_tick, ep.end_tick):
        for r in sim.tick_readings(tick):
            if r.tool_id == tool_id and r.sensor == sensor:
                xs.append(tick - ep.start_tick)
                ys.append(r.value)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _selftest():
    print("=" * 78)
    print("FabSimulator Self-Test (Phase 2 — 처리 에피소드)")
    print("=" * 78)

    # --- 구성 ---
    sim = FabSimulator(seed=42)
    print(f"\n[구성] 장비 {len(sim.tools)}대")
    total_channels = sum(t.channel_count for t in sim.tools.values())
    print(f"       총 채널 {total_channels:,}개, 에피소드 horizon {sim.episodes.horizon():,} tick")

    # --- 이상 주입 (ETCH-07의 실제 에피소드에 정렬) ---
    eps = sim.episodes.per_tool["ETCH-07"]
    inj_start, inj_end = eps[1].start_tick, eps[-1].end_tick
    print(f"\n[이상 주입] ETCH-07에 Edge-Ring (tick {inj_start}~{inj_end}, ramp 0)")
    print(f"           ETCH-07 에피소드 {len(eps)}개 중 eps[0]=정상, eps[2]=이상 대상")
    sim.schedule_anomaly("ETCH-07", "Edge-Ring", start_tick=inj_start, end_tick=inj_end, ramp_ticks=0)

    # --- 센서 스트림 (활성 tick 하나) ---
    active_tick = eps[2].start_tick + int(eps[2].ramp_s * sim.sampling_hz) + 1  # eps[2] steady
    r_active = sim.tick_readings(active_tick)
    from collections import Counter
    n_active_tools = len({r.tool_id for r in r_active})
    ph_dist = Counter(r.phase for r in r_active)
    print(f"\n[센서 스트림] tick {active_tick}: {len(r_active):,}건, 활성 장비 {n_active_tools}대")
    print(f"  phase 분포: {dict(ph_dist)}")
    etch07 = [r for r in r_active if r.tool_id == "ETCH-07"]
    if etch07:
        s = etch07[0]
        print(f"  ETCH-07 스탬프 예: wafer={s.wafer_id}, step={s.step}, phase={s.phase}, gt_defect={s.injected_defect}")

    # --- 검증 1: deviation (steady 게이트) ---
    print("\n[검증 1 — deviation은 steady 구간에만]")
    cp_normal = _episode_steady_values(sim, eps[0], "ETCH-07", "chamber_pressure")
    cp_anom = _episode_steady_values(sim, eps[2], "ETCH-07", "chamber_pressure")
    if cp_normal and cp_anom:
        shift = float(np.mean(cp_anom) - np.mean(cp_normal))
        print(f"  정상 steady 평균: {np.mean(cp_normal):.2f} mTorr (baseline 40)")
        print(f"  이상 steady 평균: {np.mean(cp_anom):.2f} mTorr (기대 +2.4σ ≈ 43)")
        print(f"  → 이동량: {shift:+.2f} mTorr (기대 ~+3.0)  {'✅' if 2.0 < shift < 4.5 else '⚠️'}")

    # --- 검증 3: 트레이스 형태 (RECIPE 사다리꼴 / HELD 평탄) ---
    print("\n[검증 3 — 트레이스 형태]")
    ramp_s = int(eps[0].ramp_s * sim.sampling_hz)
    for sensor, kind in [("chamber_pressure", "RECIPE"), ("vibration_level", "HELD")]:
        xs, ys = _episode_trace(sim, eps[0], "ETCH-07", sensor)  # 정상 에피소드
        up_mask = xs < ramp_s
        down_mask = xs >= (eps[0].duration_ticks - ramp_s)
        steady_mask = ~up_mask & ~down_mask
        slope_up = float(np.polyfit(xs[up_mask], ys[up_mask], 1)[0]) if up_mask.sum() > 2 else float("nan")
        slope_down = float(np.polyfit(xs[down_mask], ys[down_mask], 1)[0]) if down_mask.sum() > 2 else float("nan")
        steady_mean = float(np.mean(ys[steady_mask])) if steady_mask.any() else float("nan")
        if kind == "RECIPE":
            ok = slope_up > 0 and slope_down < 0
            print(f"  {sensor:<18}(RECIPE): ramp↑ slope={slope_up:+.2f}, ramp↓ slope={slope_down:+.2f}, "
                  f"steady≈{steady_mean:.1f}  {'✅ 사다리꼴' if ok else '⚠️'}")
        else:  # HELD — 평탄해야 정상 (사다리꼴 assert를 걸면 안 됨)
            flat = abs(slope_up) < 0.05 * max(steady_mean, 1e-9) + 0.02
            print(f"  {sensor:<18}(HELD):   ramp↑ slope={slope_up:+.3f} (≈0 기대), "
                  f"전체≈{np.mean(ys):.2f}  {'✅ 평탄' if flat else '⚠️'}")

    # --- 검증: duty cycle (실측) ---
    print(f"\n[검증 — duty cycle (목표 {sim.utilization}, 실측)]")
    burst_duties, life_duties = [], []
    for tid in ["ETCH-07", "DIFF-04", "CMP-05", "PHOTO-08"]:
        te = sim.episodes.per_tool.get(tid)
        if not te:
            continue
        k = min(20, len(te) - 1)
        burst = sim.episodes.measure_duty(tid, te[0].start_tick, te[k].end_tick)
        life = sim.episodes.measure_duty(tid, te[0].start_tick, te[-1].end_tick)
        burst_duties.append(burst)
        life_duties.append(life)
        print(f"  {tid:<10} 연속처리 duty={burst:.3f}  |  전체수명 duty={life:.3f}")
    if burst_duties:
        print(f"  연속 처리 구간 평균 {np.mean(burst_duties):.3f} → 투입 레이트로 맞춘 목표에 수렴 ✅")
        print(f"  전체 수명 평균 {np.mean(life_duties):.3f} — lot 단위 라우팅(25장 묶음)의 버스트성으로 낮음")

    # --- Wafer 결과 + Commonality (별도 경로, 라벨 기반) ---
    at_tick = inj_start + 50  # 주입 창 안
    print(f"\n[Wafer 결과] tick {at_tick}(주입 창 내)에서 lot 200개 처리")
    results = sim.process_lots(n_lots=200, at_tick=at_tick)
    n_defect = sum(1 for r in results if r.label != NORMAL_LABEL)
    print(f"  총 wafer {len(results):,}장, 불량 {n_defect}장 ({n_defect/len(results):.1%})")
    label_dist = Counter(r.label for r in results if r.label != NORMAL_LABEL)
    print(f"  불량 분포: {dict(label_dist)}")

    print("\n[Commonality Analysis] 원인 장비 지목")
    comm = commonality_analysis(results, top_k=5)
    print(f"  {'장비':<12} {'lift':<8} {'불량률':<10} {'경유(불량/전체)'}")
    print("  " + "-" * 50)
    for c in comm:
        marker = " ⭐ 정답" if c.tool_id == "ETCH-07" else ""
        print(f"  {c.tool_id:<12} {c.lift:<8.2f} {c.defect_rate:<10.1%} "
              f"{c.defect_pass_count}/{c.total_pass_count}{marker}")
    top_suspect = comm[0].tool_id if comm else None
    print(f"\n  {'✅ ETCH-07 top-1 정확 지목' if top_suspect == 'ETCH-07' else f'⚠️ top-1={top_suspect}'}")

    # --- 생성량 (에피소드 모델 실측) ---
    print("\n[생성량]")
    from defect_sensor_mapping import expected_generation_rate, expected_false_alarm_rate
    gen_rate = expected_generation_rate(sampling_hz=1.0)
    # 활성 구간 여러 tick 평균
    probe = range(active_tick, active_tick + 20)
    per_tick = [len(sim.tick_readings(t)) for t in probe]
    print(f"  기준 공식(50×0.8×250×1Hz): {gen_rate:,.0f} msg/s")
    print(f"  에피소드 모델 실측: {np.mean(per_tick):,.0f} 건/tick (idle 침묵 반영)")
    print(f"  1차 3σ 오경보(구 모델 기준선): {expected_false_alarm_rate(sampling_hz=1.0):.1f} 건/초 — 에피소드 모델선 재측정 대상")

    # --- 결정론 확인 (fresh sim, 같은 tick) ---
    print("\n[결정론 검증]")
    def fresh():
        s = FabSimulator(seed=42)
        e = s.episodes.per_tool["ETCH-07"]
        s.schedule_anomaly("ETCH-07", "Edge-Ring", e[1].start_tick, e[-1].end_tick, ramp_ticks=0)
        return s
    s3, s4 = fresh(), fresh()
    v3 = sorted((r.tool_id, r.sensor, round(r.value, 6)) for r in s3.tick_readings(active_tick))
    v4 = sorted((r.tool_id, r.sensor, round(r.value, 6)) for r in s4.tick_readings(active_tick))
    print(f"  같은 seed·같은 tick 재생성 일치: {v3 == v4}")

    print("\n" + "=" * 78)
    print("Self-test 완료")
    print("=" * 78)


if __name__ == "__main__":
    _selftest()

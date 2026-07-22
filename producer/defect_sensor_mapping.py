"""
FabScope — defect ↔ sensor 매핑 테이블 (Phase 2 fab simulator)
================================================================

WM-811K의 wafer defect 패턴에서 거꾸로 센서 데이터를 생성하기 위한 규칙 정의.

## 설계 원칙

1. **근거와 설계를 분리 표기한다.**
   각 규칙에 Evidence 태그를 붙여 "문헌이 보증하는 부분"과 "우리가 설계한 부분"을
   명확히 구분한다. 문헌은 [defect → 공정 모듈]까지만 보증하며, 센서 시그니처
   (어느 센서가 어느 방향으로 움직이는가)는 대부분 공정 물리에 기반한 우리 설계다.
   예외: Etch(Goodlin 2003의 19변수 + fault injection)와 CMP motor_current(US6428387).

2. **std는 Cpk 역산으로 유도한다.**
   실제 fab의 센서 변동 통계는 비공개(Goodlin 2003조차 baseline을 proprietary로 표기).
   대신 업계 양산 최소 기준인 Cpk 1.33을 가정해 규격폭에서 σ를 역산한다.
       σ = 규격폭 / (6 × 1.33) ≈ 규격폭 / 8

3. **이상 크기는 σ 단위 탐지 난이도로 배치한다.**
   목적이 "현실 재현"이 아니라 "FDC 탐지기 검증"이므로, 1차(3σ chart)가 잡는 것과
   놓치는 것이 모두 존재해야 2계층 설계를 검증할 수 있다.
   Goodlin(2003)의 주입 fault(pressure +1/+2/+3 mTorr = 0.8σ/1.6σ/2.4σ)와 같은 스케일.

4. **장비(tool) 단위로 센서가 발생한다.**
   "Etch 공정의 압력"은 없고 "ETCH-07의 압력"만 있다. 장비 ID가 없으면 센서 이상과
   wafer 불량을 이을 수 없다(commonality analysis의 존재 이유).

5. **장비 1대 = 250채널 (핵심 5~13개 + 배경 ~240개).**
   실무의 신호 대 잡음 구조 재현. 배경 채널도 정상 변동(노이즈)을 가져야 하며,
   그래야 3σ chart가 오경보를 내고 2차 탐지기의 존재 이유를 검증할 수 있다.

## Phase 1 범위 선언

- 커버: Photolithography, Etch, ThinFilm(PECVD), Diffusion, CMP, Cleaning
- 향후 확장: PVD/Sputtering, ALD, Epitaxy
  → WM-811K의 Center defect가 ESWA(2023) 기준 "thin film deposition step 실패"이므로
    Phase 1은 PECVD 커버로 충분.

## 상세 근거

Notion 16번 문서 9~15절, 20번 실무 근거 총람 참조.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal

import numpy as np

# ==============================================================================
# 0. 근거 등급 (Evidence Grade)
# ==============================================================================


class Evidence(str, Enum):
    """각 규칙/값의 근거 수준. 면접에서 '이건 어디까지 확인했나'에 정직하게 답하기 위함."""

    # 문헌이 직접 보증
    A = "A"  # 본문 전문 정독. 실험/수치가 원문에 명시됨
    B = "B"  # 원문장 대조 확인 (특허 원문, peer-reviewed 논문 초록)
    B_MINUS = "B-"  # 규격폭이 문헌에 있고, 거기서 Cpk 역산한 값
    C = "C"  # 업계 관리 관행을 가정한 뒤 역산. 가정임을 명시
    C_PLUS = "C+"  # 목적 기반 실험 설계 + 문헌 앵커 (σ-tier)
    D = "D"  # 2차 인용 (원저작 미확인)

    # 우리가 만든 것
    DESIGN = "DESIGN"  # 시뮬레이션 설계값. 문헌 근거 없음을 명시
    SCENARIO = "SCENARIO"  # 시나리오 정답지. 실측 근거가 존재할 수 없음 (민감도 분석 대상)


@dataclass(frozen=True)
class Ref:
    """근거 출처 하나."""

    grade: Evidence
    source: str  # 특허 번호, 논문명 등
    quote: str = ""  # 원문 인용 (있으면)

    def __str__(self) -> str:
        q = f' — "{self.quote}"' if self.quote else ""
        return f"[{self.grade.value}] {self.source}{q}"


# ==============================================================================
# 1. Cpk 역산 유틸
# ==============================================================================

CPK_TARGET = 1.33  # 업계 양산 승인 최소 기준. 우수 공정은 1.67


def sigma_from_spec(lsl: float, usl: float, cpk: float = CPK_TARGET) -> float:
    """규격 상/하한에서 표준편차를 역산한다.

    Cpk = (USL - LSL) / (6σ)  [중심 정렬 가정]
    →  σ = (USL - LSL) / (6 × Cpk) ≈ 규격폭 / 8   (Cpk=1.33일 때)

    예: RF power 규격 490~510 W → σ = 20 / 8 = 2.5 W

    교차 검증: SPC 관행(관리한계 = ±3σ)으로 해석해도 σ = 20/6 = 3.3 W로 같은 자릿수.
    """
    return (usl - lsl) / (6.0 * cpk)


def sigma_from_tolerance(setpoint: float, tol_pct: float, cpk: float = CPK_TARGET) -> float:
    """setpoint ± tol% 관리폭에서 σ를 역산한다. (규격이 문헌에 없을 때 — [C]등급)

    예: MFC ±1% of setpoint, 250 sccm → 규격폭 5 sccm → σ = 0.625
    """
    half_width = setpoint * tol_pct
    return sigma_from_spec(setpoint - half_width, setpoint + half_width, cpk)


# ==============================================================================
# 2. 센서 정의
# ==============================================================================

Direction = Literal["up", "down", "spike", "unstable", "any"]


class Actuation(str, Enum):
    """센서 값이 처리 에피소드 안에서 어떤 모양을 그리는가.

    처리 에피소드(wafer 1장 처리 ~60s)는 ramp-up→steady→ramp-down 사다리꼴이지만,
    모든 물리량이 이 모양을 따르는 건 아니다. '언제 레코드를 방출하나'(에피소드 통일)와
    '값이 어떤 모양이냐'(이 필드)는 별개 축이다.

    - RECIPE: 레시피로 켜지는 값. 처리 중에만 존재 → 0→setpoint→0 사다리꼴.
        RF power, 공정 가스 유량, CMP motor current, exposure_dose 등.
    - HELD:   웨이퍼 없어도 setpoint 근방 유지되거나 장비 속성 자체.
        확산로 존 온도, susceptor 온도, chamber wall temp, vibration_level,
        파티클 카운터 등 → 에피소드 내내 baseline + noise 평탄 (ramp 없음).
    """

    RECIPE = "recipe"
    HELD = "held"


@dataclass(frozen=True)
class SensorSpec:
    """핵심 센서 하나의 정상 상태 정의."""

    name: str
    baseline: float  # 정상 운영값 (setpoint)
    sigma: float  # 표준편차 (Cpk 역산 또는 관행 가정)
    unit: str
    ref: Ref  # baseline의 근거
    sigma_ref: Ref  # sigma의 근거 (별도 — 대개 등급이 다름)
    drift_per_day: float = 0.0  # 일일 드리프트 비율 (0이면 드리프트 없음)
    actuation: Actuation = Actuation.RECIPE  # 에피소드 내 값 프로파일

    @property
    def cv(self) -> float:
        """변동계수 = σ / mean. 물리량 종류별 타당성 검토용."""
        return self.sigma / self.baseline if self.baseline else 0.0


@dataclass(frozen=True)
class Deviation:
    """defect 발생 시 센서가 어떻게 벗어나는가.

    절대값이 아니라 **σ 단위**로 정의한다. baseline_std가 바뀌면 이상 크기도 함께
    스케일되므로, Cpk 역산으로 σ를 수정해도 탐지 난이도가 유지된다.
    """

    sensor: str
    mean_shift_sigma: float  # 평균 이동 (σ 배수). 음수면 하락
    std_multiplier: float = 1.0  # 변동성 배수. >1이면 흔들림 증가
    direction: Direction = "any"  # 서술용 (물리적 방향)
    ref: Ref | None = None  # 이 시그니처의 근거 (없으면 우리 설계)

    def apply(self, spec: SensorSpec) -> tuple[float, float]:
        """이상 상태의 (mean, std)를 계산한다."""
        mean = spec.baseline + self.mean_shift_sigma * spec.sigma
        std = spec.sigma * self.std_multiplier
        return mean, std


# ==============================================================================
# 3. 탐지 난이도 tier (σ-tier DoE)
# ==============================================================================


class DetectionTier(str, Enum):
    """이상 크기를 탐지 난이도로 배치한다.

    목적: FDC 2계층(1차 3σ chart + 2차 Autoencoder)을 검증하려면
    "1차가 잡는 것"과 "1차가 놓치고 2차만 잡는 것"이 모두 있어야 한다.

    앵커: Goodlin(2003)의 주입 fault
      - pressure +1/+2/+3 mTorr → σ=1.25 환산 시 0.8σ / 1.6σ / 2.4σ  (경계~표준)
      - RF ±8~12 W → σ=2.5 환산 시 3.2σ / 4.8σ  (표준~명백)
    실제 fault injection 연구도 탐지 경계를 걸치도록 크기를 설계했다.
    """

    OBVIOUS = "obvious"  # 4~6σ — 1차가 즉시 탐지해야 함
    STANDARD = "standard"  # 2~3σ — 1차가 탐지하되 수 샘플 소요
    BORDERLINE = "borderline"  # 1~2σ — 1차가 놓치기 시작
    HIDDEN = "hidden"  # 이동 ~0σ + std만 증가 — 1차 무력, 2차만 탐지


# ==============================================================================
# 4. 배경 채널 원형 (Channel Archetypes)
# ==============================================================================
#
# 장비당 250채널 중 ~240개는 defect와 무관한 배경 채널이다.
# 이들을 개별 정의할 수는 없으므로 물리량 종류별 원형에서 생성한다.
#
# ## 왜 mean은 개별 정당화가 불필요한가
#
# FDC는 raw 값이 아니라 z-score `(x - mean) / std`로 감시한다. 3σ chart도 AE도
# 정규화된 값을 본다. 즉 heater_zone_07의 mean이 350°C냐 420°C냐는 탐지 성능에
# 영향이 없다. 중요한 건 **std 대비 얼마나 벗어나는가**뿐.
#
# → 그래서 진짜 근거가 필요한 건 **CV(변동계수 = std/mean)**이고,
#   이것만 물리량 종류별로 대면 된다.
#
# ## 저유량 보정 (MFC 정밀도 조사에서 나온 통찰)
#
# MFC 정밀도 사양에는 두 방식이 있다.
#   - "% of setpoint": 설정값의 ±1% (MKS P4B/P9B)
#   - "% of full scale": 최대 유량의 ±1% → **저유량에서 상대오차가 10배까지 커진다**
#     원문: "For a 100 sccm full scale controller with 1% FS accuracy... at 10 sccm
#            setpoint, the same 1 sccm error is ±10%"
# → CV를 상수로 두면 비현실적. mean에 반비례하는 성분을 추가한다.

ARCHETYPE_TEMPERATURE = "temperature"
ARCHETYPE_GAS_FLOW = "gas_flow"
ARCHETYPE_PRESSURE = "pressure"
ARCHETYPE_POWER = "power"
ARCHETYPE_VALVE = "valve_position"
ARCHETYPE_PARTICLE = "particle_count"


@dataclass(frozen=True)
class ChannelArchetype:
    """배경 채널의 물리량 종류별 생성 규칙."""

    name: str
    mean_range: tuple[float, float]
    unit: str
    cv_base: float  # 기본 변동계수
    cv_low_flow_term: float = 0.0  # 저유량 보정: cv = cv_base + term/mean
    distribution: Literal["normal", "poisson"] = "normal"
    drift_per_day: float = 0.0
    actuation: Actuation = Actuation.RECIPE  # 이 원형의 값 프로파일
    ref: Ref = field(default_factory=lambda: Ref(Evidence.DESIGN, "설계값"))

    def sample_spec(self, rng: np.random.Generator, index: int) -> SensorSpec:
        """이 원형에서 배경 채널 하나를 생성한다."""
        mean = float(rng.uniform(*self.mean_range))

        if self.distribution == "poisson":
            # 카운트 데이터는 std = √mean (포아송 분포의 성질)
            sigma = float(np.sqrt(mean))
        else:
            cv = self.cv_base + (self.cv_low_flow_term / mean if mean else 0.0)
            sigma = mean * cv

        return SensorSpec(
            name=f"{self.name}_{index:03d}",
            baseline=mean,
            sigma=sigma,
            unit=self.unit,
            ref=self.ref,
            sigma_ref=self.ref,
            drift_per_day=self.drift_per_day,
            actuation=self.actuation,
        )


CHANNEL_ARCHETYPES: dict[str, ChannelArchetype] = {
    ARCHETYPE_TEMPERATURE: ChannelArchetype(
        name=ARCHETYPE_TEMPERATURE,
        mean_range=(25.0, 400.0),
        unit="C",
        cv_base=0.003,  # 0.3%
        drift_per_day=0.002,  # 히터 열화
        actuation=Actuation.HELD,  # 챔버/히터 온도는 상시 유지 (열질량)
        ref=Ref(
            Evidence.C,
            "US8501499 + hotplate 정밀도",
            "챔버 온도 30~90C, wafer holder 25~35C 같은 좁은 관리폭. hotplate ±1C",
        ),
    ),
    ARCHETYPE_GAS_FLOW: ChannelArchetype(
        name=ARCHETYPE_GAS_FLOW,
        mean_range=(5.0, 500.0),
        unit="sccm",
        cv_base=0.01,  # 1% of setpoint (MKS P4B/P9B 사양)
        cv_low_flow_term=0.5,  # 저유량 보정 → cv = 0.01 + 0.5/mean
        drift_per_day=0.005,  # MFC 드리프트 (연 1회 캘리브레이션 권장)
        ref=Ref(
            Evidence.B,
            "MKS P4B/P9B, Semiconductor Digest",
            "±1% of set point. 'Today, 1% accuracy is required for challenging applications'",
        ),
    ),
    ARCHETYPE_PRESSURE: ChannelArchetype(
        name=ARCHETYPE_PRESSURE,
        mean_range=(5.0, 100.0),
        unit="mTorr",
        cv_base=0.015,  # 1.5%
        drift_per_day=0.003,
        actuation=Actuation.HELD,  # base pressure는 idle에도 존재
        ref=Ref(
            Evidence.B_MINUS,
            "US8501499 TEOS OE 35~45 mTorr",
            "규격폭 10 mTorr / 중심 40 → Cpk 1.33 역산 시 CV ≈ 1.5%",
        ),
    ),
    ARCHETYPE_POWER: ChannelArchetype(
        name=ARCHETYPE_POWER,
        mean_range=(50.0, 700.0),
        unit="W",
        cv_base=0.005,  # 0.5%
        drift_per_day=0.001,  # RF 매칭 열화
        ref=Ref(
            Evidence.B_MINUS,
            "FDC 실무 가이드 RF forward 490~510 W",
            "규격폭 20 W → σ=2.5 → CV 0.5%",
        ),
    ),
    ARCHETYPE_VALVE: ChannelArchetype(
        name=ARCHETYPE_VALVE,
        mean_range=(10.0, 90.0),
        unit="%",
        cv_base=0.03,  # 3% — 기계식 액추에이터는 전자식보다 반복 정밀도가 낮다
        drift_per_day=0.004,
        ref=Ref(
            Evidence.C,
            "Goodlin(2003) Table 1의 vat valve position",
            "감시 변수로 실재하나 정밀도 사양은 미확인. 기계식 특성으로 3% 가정",
        ),
    ),
    ARCHETYPE_PARTICLE: ChannelArchetype(
        name=ARCHETYPE_PARTICLE,
        mean_range=(1.0, 50.0),
        unit="count",
        cv_base=0.0,  # 포아송이라 CV를 쓰지 않음
        distribution="poisson",
        drift_per_day=0.01,  # 챔버 오염 누적
        actuation=Actuation.HELD,  # 파티클 카운터는 상시 계측
        ref=Ref(
            Evidence.B,
            "포아송 분포의 통계적 성질",
            "카운트 데이터는 std = sqrt(mean). mean=5면 std≈2.2 (CV 45%)",
        ),
    ),
}


# 모듈별 배경 채널 구성 — 물리적으로 그럴듯한 비율
# Etch/ThinFilm은 가스·RF가 많고, CMP는 압력·유체·파티클, Diffusion은 온도(다수 zone),
# Cleaning은 파티클 중심.
BACKGROUND_CHANNEL_MIX: dict[str, dict[str, int]] = {
    "Etch": {
        ARCHETYPE_TEMPERATURE: 55,
        ARCHETYPE_GAS_FLOW: 50,
        ARCHETYPE_PRESSURE: 30,
        ARCHETYPE_POWER: 25,
        ARCHETYPE_VALVE: 40,
        ARCHETYPE_PARTICLE: 37,
    },
    "ThinFilm": {
        ARCHETYPE_TEMPERATURE: 75,
        ARCHETYPE_GAS_FLOW: 60,
        ARCHETYPE_PRESSURE: 25,
        ARCHETYPE_POWER: 30,
        ARCHETYPE_VALVE: 30,
        ARCHETYPE_PARTICLE: 17,
    },
    "CMP": {
        ARCHETYPE_TEMPERATURE: 40,
        ARCHETYPE_GAS_FLOW: 20,
        ARCHETYPE_PRESSURE: 58,
        ARCHETYPE_POWER: 10,
        ARCHETYPE_VALVE: 50,
        ARCHETYPE_PARTICLE: 61,
    },
    "Photolithography": {
        ARCHETYPE_TEMPERATURE: 70,
        ARCHETYPE_GAS_FLOW: 30,
        ARCHETYPE_PRESSURE: 20,
        ARCHETYPE_POWER: 40,
        ARCHETYPE_VALVE: 45,
        ARCHETYPE_PARTICLE: 40,
    },
    "Diffusion": {
        ARCHETYPE_TEMPERATURE: 118,
        ARCHETYPE_GAS_FLOW: 40,
        ARCHETYPE_PRESSURE: 20,
        ARCHETYPE_POWER: 15,
        ARCHETYPE_VALVE: 30,
        ARCHETYPE_PARTICLE: 20,
    },
    "Cleaning": {
        ARCHETYPE_TEMPERATURE: 30,
        ARCHETYPE_GAS_FLOW: 40,
        ARCHETYPE_PRESSURE: 40,
        ARCHETYPE_POWER: 15,
        ARCHETYPE_VALVE: 40,
        ARCHETYPE_PARTICLE: 80,
    },
}

# 장비당 총 채널 수. 문헌 범위(장비당 수백~1,000개) 내에서 선택한 설계값.
#   - 첨단 장비: 1,000개+ 변수 중 도메인 전문가가 30개 선별 (arXiv:2606.00923)
#   - SECOM 590개는 '라인 전체' 집계라 층위가 다름 (장비당 아님)
#   - Goodlin(2003)은 etcher 1대에서 핵심 19변수 감시
# 우리 총 채널: 250 × 50대 = 12,500 (SECOM보다 21배 많음)
CHANNELS_PER_TOOL = 250

# 결측률 — SECOM이 결측 4.5%, 28개 센서는 상습 미보고
MISSING_RATE = 0.05


# ==============================================================================
# 5. 공정 모듈별 핵심 센서 baseline
# ==============================================================================
#
# 각 센서의 baseline(정상 운영값)은 특허·업계 가이드의 실무 범위에서 가져왔다.
# 상당수가 실제 공개 레시피와 정확히 일치한다:
#   - bake_temp 110°C     → 실제 클린룸 레시피(AZ 9260/5214E) 두 건 모두 110°C
#   - furnace_temp 1000°C → US8153538 "held at about 1000°C for 30 minutes"
#   - ramp_rate 10°C/min  → US8153538 실제 레시피
#   - backside_he 15 Torr → US8501499 center 15~25 Torr
#   - susceptor_temp 400°C, rf_power_hf 500W, pressure 1.5 Torr → US6054735 Novellus 레시피
#
# sigma는 대부분 Cpk 1.33 역산. 규격폭이 문헌에 있으면 [B-], 관행 가정이면 [C].

PROCESS_BASELINES: dict[str, dict[str, SensorSpec]] = {}


# ------------------------------------------------------------------------------
# Etch — Edge-Ring, Loc, Near-full의 원인 공정
# ------------------------------------------------------------------------------
# 근거 최상급. Goodlin(2003)이 Lam 9600 etcher에서 19개 tool-state 변수를 감시하고
# pressure/RF fault를 실제 주입한 실험이 있어, [모듈→센서→방향→크기] 전 사슬이 확보됨.
#
# US8501499(Adaptive recipe selector)가 실제 레시피의 파라미터별 운영 범위를 명시:
#   TEOS OE etch: pressure 35~45 mTorr, top power 550~650 W, lower power 90~110 W,
#                 CF4 40~60 sccm, CHF3 40~60 sccm, O2 3~7 sccm,
#                 backside He center 15~25 Torr / edge 27~33 Torr
#
# [결정] chamber_pressure를 80 → 40 mTorr로 조정.
#   US4889609: "20 mTorr to 200 mTorr and preferably 40 mTorr"
#   US6903023: 구체 레시피에서 pressure 40 mTorr
#   US8501499: 35~45 mTorr
#   → 80은 US6583064의 "below about 80 mTorr"(상한)에 걸치는 값이었으나,
#     특허들이 일관되게 "preferably 40"을 지목하므로 40이 더 전형적.

_etch_pressure_sigma = sigma_from_spec(35.0, 45.0)  # US8501499 규격폭 → σ = 1.25
_etch_rf_top_sigma = sigma_from_spec(550.0, 650.0)  # US8501499 → σ = 12.5
_etch_rf_bias_sigma = sigma_from_spec(90.0, 110.0)  # US8501499 → σ = 2.5

PROCESS_BASELINES["Etch"] = {
    "chamber_pressure": SensorSpec(
        name="chamber_pressure",
        baseline=40.0,
        sigma=_etch_pressure_sigma,  # 1.25
        unit="mTorr",
        ref=Ref(
            Evidence.B,
            "US4889609 / US6903023 / US8501499",
            "'20~200 mTorr and preferably 40 mTorr'. US8501499 TEOS OE는 35~45 mTorr",
        ),
        sigma_ref=Ref(
            Evidence.B_MINUS,
            "US8501499 규격폭 35~45 → Cpk 1.33 역산",
            "σ = 10 / 8 = 1.25 mTorr",
        ),
    ),
    # RF를 top/bias로 분리 — US8501499가 별도 관리
    "rf_power_top": SensorSpec(
        name="rf_power_top",
        baseline=600.0,
        sigma=_etch_rf_top_sigma,  # 12.5
        unit="W",
        ref=Ref(Evidence.B, "US8501499", "top power 550~650 W (TEOS OE), 600~700 W (BT)"),
        sigma_ref=Ref(Evidence.B_MINUS, "US8501499 규격폭 100 W → Cpk 역산", "σ = 100/8 = 12.5"),
        drift_per_day=0.001,  # RF 매칭 열화
    ),
    "rf_power_bias": SensorSpec(
        name="rf_power_bias",
        baseline=100.0,
        sigma=_etch_rf_bias_sigma,  # 2.5
        unit="W",
        ref=Ref(Evidence.B, "US8501499", "lower power 90~110 W"),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 20 W → Cpk 역산", "σ = 20/8 = 2.5"),
    ),
    # 가스별 유량 분리 — US8501499가 CF4/CHF3/O2를 각각 관리
    "gas_flow_cf4": SensorSpec(
        name="gas_flow_cf4",
        baseline=50.0,
        sigma=sigma_from_spec(40.0, 60.0),  # 2.5
        unit="sccm",
        ref=Ref(Evidence.B, "US8501499", "CF4 40~60 sccm"),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 20 sccm → Cpk 역산", "σ = 2.5"),
        drift_per_day=0.005,  # MFC 드리프트
    ),
    "gas_flow_chf3": SensorSpec(
        name="gas_flow_chf3",
        baseline=50.0,
        sigma=sigma_from_spec(40.0, 60.0),  # 2.5
        unit="sccm",
        ref=Ref(Evidence.B, "US8501499", "CHF3 40~60 sccm"),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 20 sccm → Cpk 역산", "σ = 2.5"),
        drift_per_day=0.005,
    ),
    "gas_flow_o2": SensorSpec(
        name="gas_flow_o2",
        baseline=5.0,
        sigma=sigma_from_spec(3.0, 7.0),  # 0.5
        unit="sccm",
        ref=Ref(Evidence.B, "US8501499", "O2 3~7 sccm"),
        sigma_ref=Ref(
            Evidence.B_MINUS,
            "규격폭 4 sccm → Cpk 역산. 저유량이라 CV 10%로 큼(MFC %FS 특성과 부합)",
            "σ = 0.5",
        ),
        drift_per_day=0.005,
    ),
    # backside He를 center/edge로 분리 — US8501499가 따로 제어
    # ⭐ 이 분리 자체가 "wafer 중심/가장자리 불균일이 실제 제어 대상"임을 보여줌
    "backside_he_center": SensorSpec(
        name="backside_he_center",
        baseline=20.0,
        sigma=sigma_from_spec(15.0, 25.0),  # 1.25
        unit="Torr",
        ref=Ref(Evidence.B, "US8501499 / US5933759", "center 15~25 Torr / 10~17 Torr"),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 10 Torr → Cpk 역산", "σ = 1.25"),
    ),
    "backside_he_edge": SensorSpec(
        name="backside_he_edge",
        baseline=30.0,
        sigma=sigma_from_spec(27.0, 33.0),  # 0.75
        unit="Torr",
        ref=Ref(Evidence.B, "US8501499", "edge 27~33 Torr — center와 별도 제어"),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 6 Torr → Cpk 역산", "σ = 0.75"),
    ),
    # Goodlin Table 1의 감시 변수 (throttle_valve의 실무 명칭)
    "vat_valve_position": SensorSpec(
        name="vat_valve_position",
        baseline=50.0,
        sigma=sigma_from_tolerance(50.0, 0.08),  # ±8% → σ = 1.0
        unit="%",
        ref=Ref(
            Evidence.B,
            "Goodlin(2003) Table 1",
            "Lam 9600의 19개 tool-state 변수 중 'Vat Valve Position' 실재",
        ),
        sigma_ref=Ref(Evidence.C, "기계식 액추에이터 ±8% 가정", "정밀도 사양 미확인"),
        drift_per_day=0.004,
    ),
    # ⭐ [결정] vibration_level 추가 — Loc의 최약 고리 해소
    # Hansen & Thyregod(1998): Loc은 "excessive vibration of a specific machine"이 원인
    # 기존에는 진동을 측정할 센서가 없어 pressure/RF의 std 증가로 간접 표현했으나,
    # "진동 → 챔버 압력 변동성 증가"는 미확인 추론이었다.
    # 진동 센서를 추가하면 인과가 직결되어 논거가 단순해진다.
    "vibration_level": SensorSpec(
        name="vibration_level",
        baseline=0.5,
        sigma=sigma_from_tolerance(0.5, 0.20),  # ±20% → σ = 0.025
        unit="mm/s_rms",
        ref=Ref(
            Evidence.C,
            "Hansen & Thyregod(1998) via ESWA/Frontiers",
            "Loc의 원인이 'excessive vibration'. 진동 센서는 회전기계 모니터링의 표준 계측",
        ),
        sigma_ref=Ref(Evidence.DESIGN, "설계값", "진동 baseline 통계는 문헌 미확보"),
        drift_per_day=0.003,  # 베어링 마모
    ),
}


# ------------------------------------------------------------------------------
# ThinFilm (PECVD) — Center의 원인 공정
# ------------------------------------------------------------------------------
# ESWA(2023): "Center is caused by the failure of the thin film deposition step"
#
# [결정] rf_power 600 → 500 W, chamber_pressure 5 → 1.5 Torr
#   US6054735(Novellus Concept One)가 실제 레시피를 허용 범위까지 명시:
#     RF power 500 W (허용 480~520), pressure 1.5 Torr (허용 1.4~1.6),
#     SiH4 70 sccm (허용 65~75), N2O 4000 sccm (허용 3900~4100), temp 400°C
#   → 허용 범위가 있으므로 Cpk 역산으로 σ를 직접 유도할 수 있다. 근거가 가장 단단한 모듈.
#
# ⭐ Center defect의 물리적 근거 (US4892753 / US5000113, Applied Materials):
#   "uniform susceptor and wafer temperatures, including both absolute temperature
#    uniformity and **spatial uniformity across the susceptor/wafer**;
#    **uniform gas flow distribution across the wafer**"
#   "**uniform radial pumping** provides uniform gas flow across the wafer"
#   → 장비 설계의 핵심 과제가 '반경 방향 균일성'이며, 이것이 깨지면 중심부에 결함.
#     Diffusion의 RTD(Radial Thermal Delta)와 대칭 구조.

PROCESS_BASELINES["ThinFilm"] = {
    # ⭐ susceptor 온도를 center/edge로 분리 — Center defect의 직접 측정
    "susceptor_temp_center": SensorSpec(
        name="susceptor_temp_center",
        baseline=400.0,
        sigma=sigma_from_tolerance(400.0, 0.0125),  # ±5°C → σ = 1.25
        unit="C",
        ref=Ref(
            Evidence.B,
            "US6054735 (Novellus Concept One) / US8987039 (AMAT)",
            "deposition temperature 400°C. AMAT DxZ는 susceptor 100~400°C",
        ),
        sigma_ref=Ref(Evidence.C, "±5°C 관리폭 가정 → Cpk 역산", "σ = 1.25"),
        drift_per_day=0.002,
    ),
    "susceptor_temp_edge": SensorSpec(
        name="susceptor_temp_edge",
        baseline=400.0,
        sigma=sigma_from_tolerance(400.0, 0.0125),  # 1.25
        unit="C",
        ref=Ref(
            Evidence.B,
            "US4892753 / US5000113",
            "'spatial uniformity across the susceptor/wafer' — center/edge 균일성이 설계 과제",
        ),
        sigma_ref=Ref(Evidence.C, "±5°C 관리폭 가정", "σ = 1.25"),
        drift_per_day=0.002,
    ),
    # RF를 HF/LF로 분리 — US9598771("Dielectric film defect reduction")
    "rf_power_hf": SensorSpec(
        name="rf_power_hf",
        baseline=500.0,
        sigma=sigma_from_spec(480.0, 520.0),  # 5.0
        unit="W",
        ref=Ref(
            Evidence.B,
            "US6054735 (Novellus Concept One)",
            "'RF power is reduced to about 500 watts although power in a range from "
            "480 watts to 520 watts is suitable'",
        ),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 40 W 명시 → Cpk 역산", "σ = 40/8 = 5.0"),
        drift_per_day=0.001,
    ),
    "rf_power_lf": SensorSpec(
        name="rf_power_lf",
        baseline=300.0,
        sigma=sigma_from_tolerance(300.0, 0.04),  # ±4% → σ = 3.0
        unit="W",
        ref=Ref(
            Evidence.B,
            "US9598771 'Dielectric film defect reduction'",
            "high frequency RF 450 W + low frequency RF 300 W를 별도 제어",
        ),
        sigma_ref=Ref(Evidence.C, "±4% 관리폭 가정", "σ = 3.0"),
    ),
    "chamber_pressure": SensorSpec(
        name="chamber_pressure",
        baseline=1500.0,  # 1.5 Torr = 1500 mTorr
        sigma=sigma_from_spec(1400.0, 1600.0),  # 25.0 mTorr
        unit="mTorr",
        ref=Ref(
            Evidence.B,
            "US6054735",
            "'Pressure is applied at about 1.5 torr although pressures from "
            "1.4 torr to 1.6 torr are suitable'",
        ),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 0.2 Torr 명시 → Cpk 역산", "σ = 25 mTorr"),
    ),
    "precursor_flow_sih4": SensorSpec(
        name="precursor_flow_sih4",
        baseline=70.0,
        sigma=sigma_from_spec(65.0, 75.0),  # 1.25
        unit="sccm",
        ref=Ref(
            Evidence.B,
            "US6054735",
            "'silane flow rate is reduced to about 70 sccm, although flow rates "
            "from 65 sccm to 75 sccm are suitable'",
        ),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 10 sccm 명시 → Cpk 역산", "σ = 1.25"),
        drift_per_day=0.005,
    ),
    "oxidant_flow_n2o": SensorSpec(
        name="oxidant_flow_n2o",
        baseline=4000.0,
        sigma=sigma_from_spec(3900.0, 4100.0),  # 25.0
        unit="sccm",
        ref=Ref(
            Evidence.B,
            "US6054735",
            "'nitrous oxide flow rate is reduced to about 4000 sccm, although "
            "flow rates from 3900 sccm to 4100 sccm are suitable'",
        ),
        sigma_ref=Ref(Evidence.B_MINUS, "규격폭 200 sccm 명시 → Cpk 역산", "σ = 25"),
        drift_per_day=0.005,
    ),
    # ⭐ lamp_power — US6040022 "with the aid of the lamp module",
    #    arXiv:2606.00923의 실제 감시 변수 목록에 "lamp power" 포함
    "lamp_power": SensorSpec(
        name="lamp_power",
        baseline=800.0,
        sigma=sigma_from_tolerance(800.0, 0.02),  # ±2% → σ = 4.0
        unit="W",
        ref=Ref(
            Evidence.B,
            "US6040022 / arXiv:2606.00923",
            "'wafer temperature provided with the aid of the lamp module'. "
            "실제 생산라인 VM 연구의 감시 변수에 'lamp power' 포함",
        ),
        sigma_ref=Ref(Evidence.C, "±2% 관리폭 가정", "σ = 4.0"),
        drift_per_day=0.002,  # 램프 열화
    ),
    "deposition_rate": SensorSpec(
        name="deposition_rate",
        baseline=0.55,  # 5500 Å/min = 0.55 μm/min
        sigma=sigma_from_tolerance(0.55, 0.03),  # ±3% → σ = 0.021
        unit="um/min",
        ref=Ref(
            Evidence.B,
            "US6054735",
            "표준 공정에서 'a deposition rate of 5500 angstroms per minute is achieved'",
        ),
        sigma_ref=Ref(Evidence.C, "±3% 관리폭 가정", "σ ≈ 0.02"),
        drift_per_day=0.008,  # 챔버 벽 오염 누적 → 세정 시 리셋
    ),
    "electrode_spacing": SensorSpec(
        name="electrode_spacing",
        baseline=7.62,  # 300 mils
        sigma=sigma_from_tolerance(7.62, 0.01),  # ±1% → σ = 0.019
        unit="mm",
        ref=Ref(
            Evidence.B,
            "US6040022 / EPJ Photovoltaics",
            "electrode spacing d=300 mils. EPJ는 susceptor 수직 이동으로 5~75mm 가변",
        ),
        sigma_ref=Ref(Evidence.C, "±1% 가정", "기계적 위치 제어"),
    ),
    "chamber_wall_temp": SensorSpec(
        name="chamber_wall_temp",
        baseline=80.0,
        sigma=sigma_from_tolerance(80.0, 0.025),  # ±2.5% → σ = 0.25
        unit="C",
        ref=Ref(
            Evidence.B,
            "EPJ Photovoltaics (2014)",
            "'All chamber walls are heated to 80 °C'",
        ),
        sigma_ref=Ref(Evidence.C, "±2°C 관리폭 가정", "σ = 0.25"),
        drift_per_day=0.001,
    ),
    "chamber_clean_cycle_count": SensorSpec(
        name="chamber_clean_cycle_count",
        baseline=50.0,  # 마지막 세정 이후 처리한 wafer 수
        sigma=0.0,  # 카운터라 노이즈 없음 (단조 증가 후 리셋)
        unit="count",
        ref=Ref(
            Evidence.B,
            "EPJ Photovoltaics",
            "'The chambers are usually cleaned after each layer deposition using a "
            "NF3/Ar etch plasma followed by a conditioning layer to ensure stable and "
            "reproducible process start conditions and reduce contamination'",
        ),
        sigma_ref=Ref(Evidence.DESIGN, "카운터", "드리프트 리셋 트리거"),
    ),
}


# ------------------------------------------------------------------------------
# Diffusion (Furnace) — Edge-Loc의 원인 공정
# ------------------------------------------------------------------------------
# Hansen, Nair & Friedman(1997) via ESWA/Frontiers:
#   "Edge-Loc is caused by uneven heating during diffusion"
#
# ⭐ US7256370이 이 문제에 전용 용어를 부여했다:
#   "non-uniform heating of substrates from a substrate **center to an outer edge**
#    (this non-uniform heating is referred to as **'RTD' or Radial Thermal Delta**)"
#   → 불균일 가열이 diffusion furnace의 알려진 고질 문제이며, 업계 전용 용어까지 존재.
#     2차 인용(Hansen)에만 의존하던 근거가 특허 원문으로 격상됨.
#
# ⚠️ 기존 설계의 최대 결함:
#   zone1/2/3(수직 방향 heater zone)만 있어 **반경 방향 온도차를 측정할 센서가 없었다.**
#   Edge-Loc의 물리적 원인을 직접 측정하지 못하는 상태였음.
#   → wafer_center_temp / wafer_edge_temp 추가로 해소.

PROCESS_BASELINES["Diffusion"] = {
    # ⭐ RTD의 직접 측정 — 반경 방향 온도
    "wafer_center_temp": SensorSpec(
        name="wafer_center_temp",
        baseline=1000.0,
        sigma=sigma_from_tolerance(1000.0, 0.005),  # ±5°C → σ = 1.25
        unit="C",
        ref=Ref(
            Evidence.B,
            "US10388762 / US8153538",
            "어닐링 'preferably 1000~1200°C'. 실제 레시피 'held at about 1000°C for 30 min'",
        ),
        sigma_ref=Ref(Evidence.C, "±5°C 균일성 스펙 가정 → Cpk 역산", "σ = 1.25"),
        drift_per_day=0.001,
    ),
    "wafer_edge_temp": SensorSpec(
        name="wafer_edge_temp",
        baseline=1000.0,
        sigma=sigma_from_tolerance(1000.0, 0.005),  # 1.25
        unit="C",
        ref=Ref(
            Evidence.B,
            "US7256370 (RTD)",
            "'non-uniform heating from a substrate center to an outer edge'",
        ),
        sigma_ref=Ref(Evidence.C, "±5°C 가정", "σ = 1.25"),
        drift_per_day=0.001,
    ),
    # 수직 방향 heater zone (기존 유지)
    "heater_zone_top": SensorSpec(
        name="heater_zone_top",
        baseline=1000.0,
        sigma=sigma_from_tolerance(1000.0, 0.005),
        unit="C",
        ref=Ref(Evidence.B, "US10388762", "어닐링 900~1200°C"),
        sigma_ref=Ref(Evidence.C, "±5°C 가정", "σ = 1.25"),
        drift_per_day=0.001,
    ),
    "heater_zone_center": SensorSpec(
        name="heater_zone_center",
        baseline=1000.0,
        sigma=sigma_from_tolerance(1000.0, 0.005),
        unit="C",
        ref=Ref(Evidence.B, "US10388762", "어닐링 900~1200°C"),
        sigma_ref=Ref(Evidence.C, "±5°C 가정", "σ = 1.25"),
        drift_per_day=0.001,
    ),
    "heater_zone_bottom": SensorSpec(
        name="heater_zone_bottom",
        baseline=1000.0,
        sigma=sigma_from_tolerance(1000.0, 0.005),
        unit="C",
        ref=Ref(Evidence.B, "US10388762", "어닐링 900~1200°C"),
        sigma_ref=Ref(Evidence.C, "±5°C 가정", "σ = 1.25"),
        drift_per_day=0.001,
    ),
    "ramp_rate": SensorSpec(
        name="ramp_rate",
        baseline=10.0,
        sigma=sigma_from_tolerance(10.0, 0.05),  # ±5% → σ = 0.0625
        unit="C/min",
        ref=Ref(
            Evidence.B,
            "US8153538 / US8759198",
            "실제 레시피 'ramp rate of 10 °C/min'. 가속 램프는 5.5~11 °C/min",
        ),
        sigma_ref=Ref(Evidence.C, "±5% 가정", "σ ≈ 0.06"),
    ),
    "n2_flow": SensorSpec(
        name="n2_flow",
        baseline=5000.0,
        sigma=sigma_from_tolerance(5000.0, 0.01),  # MFC ±1% → σ = 6.25
        unit="sccm",
        ref=Ref(Evidence.B, "US10388762", "'N2 gas ranges from 100 sccm to 10000 sccm'"),
        sigma_ref=Ref(
            Evidence.B_MINUS,
            "MFC ±1% of setpoint (MKS P4B/P9B) → Cpk 역산",
            "σ = 100/8 = 12.5 → 보수적으로 6.25",
        ),
        drift_per_day=0.005,
    ),
}


# ------------------------------------------------------------------------------
# CMP — Scratch의 원인 공정
# ------------------------------------------------------------------------------
# Wang & Bensmial(2006) via ESWA/Frontiers:
#   "a scratch is a result of machine handling problems"
#
# ⭐ motor_current는 실무 표준 감시 기법이다 (근거가 가장 단단한 센서 중 하나):
#   US6428387: "the friction between the pad and the wafer carrier increases...
#               more current is supplied to the motor... **This increase in wafer carrier
#               motor current can be monitored** and used to determine the end point"
#   JEES: "the friction force changes, causing a detectable shift in motor current"
#   → "pad 이상/입자 혼입 → 마찰 증가 → 전류 스파이크"는 우리가 지어낸 게 아니라
#     실제 fab이 endpoint 검출에 쓰는 물리 원리.
#
# ⭐ JEES가 Scratch의 직접 원인도 명시:
#   "harder pads (such as IC1000) deliver better global planarization efficiency but
#    **can increase micro-scratch defects**. Softer pads reduce defects but sacrifice
#    planarization"
#   → pad 경도/glazing이 micro-scratch를 만든다. pad 상태 센서가 필요.
#
# US9777192(Strasbaugh nSpire 표준 레시피)는 3개 압력을 동시 제어:
#   down 2.0 psi / backside 0.5 psi / retaining ring 2.5 psi, table·carrier 95/86 rpm,
#   slurry 200 ml/min, conditioner 4.0 lbs, pad IC1000

PROCESS_BASELINES["CMP"] = {
    "polish_pressure": SensorSpec(
        name="polish_pressure",
        baseline=3.0,
        sigma=sigma_from_tolerance(3.0, 0.10),  # ±10% → σ = 0.075
        unit="psi",
        ref=Ref(
            Evidence.B,
            "JEES CMP 기술가이드 / US9777192 / US6705928",
            "production 운영 범위 1~6 psi. 표준 레시피 down pressure 2.0 psi",
        ),
        sigma_ref=Ref(Evidence.C, "±10% 관리폭 가정", "σ = 0.075"),
    ),
    # US9777192의 3압력 중 나머지 둘
    "backside_pressure": SensorSpec(
        name="backside_pressure",
        baseline=0.5,
        sigma=sigma_from_tolerance(0.5, 0.10),  # 0.0125
        unit="psi",
        ref=Ref(Evidence.B, "US9777192", "back side pressure: 0.5 psi (34.5 mbar)"),
        sigma_ref=Ref(Evidence.C, "±10% 가정", "σ ≈ 0.013"),
    ),
    "retaining_ring_pressure": SensorSpec(
        name="retaining_ring_pressure",
        baseline=2.5,
        sigma=sigma_from_tolerance(2.5, 0.10),  # 0.0625
        unit="psi",
        ref=Ref(Evidence.B, "US9777192", "retaining ring pressure: 2.5 psi (172 mbar)"),
        sigma_ref=Ref(Evidence.C, "±10% 가정", "σ ≈ 0.063"),
    ),
    "platen_speed": SensorSpec(
        name="platen_speed",
        baseline=95.0,
        sigma=sigma_from_tolerance(95.0, 0.03),  # ±3% → σ = 0.594
        unit="rpm",
        ref=Ref(
            Evidence.B,
            "US9777192 / JEES",
            "polishing table speed 95 rpm. 운영 범위 30~100 rpm",
        ),
        sigma_ref=Ref(Evidence.C, "±3% 가정 (회전 제어)", "σ ≈ 0.6"),
    ),
    "carrier_speed": SensorSpec(
        name="carrier_speed",
        baseline=86.0,
        sigma=sigma_from_tolerance(86.0, 0.03),  # 0.538
        unit="rpm",
        ref=Ref(Evidence.B, "US9777192", "carrier speed 86 rpm"),
        sigma_ref=Ref(Evidence.C, "±3% 가정", "σ ≈ 0.54"),
    ),
    "slurry_flow": SensorSpec(
        name="slurry_flow",
        baseline=200.0,
        sigma=sigma_from_tolerance(200.0, 0.05),  # ±5% → σ = 1.25
        unit="ml/min",
        ref=Ref(
            Evidence.B,
            "US9777192 / JEES",
            "slurry flow rate 200 ml/min. 운영 범위 100~300 ml/min",
        ),
        sigma_ref=Ref(Evidence.C, "±5% 가정 (액체 펌프)", "σ = 1.25"),
        drift_per_day=0.004,
    ),
    # ⭐ 실무 표준 감시 기법 — Scratch의 핵심 시그니처
    "motor_current": SensorSpec(
        name="motor_current",
        baseline=12.0,
        sigma=sigma_from_tolerance(12.0, 0.02),  # ±2% → σ = 0.03
        unit="A",
        ref=Ref(
            Evidence.B,
            "US6428387 / JEES",
            "'This increase in wafer carrier motor current can be monitored and used "
            "to determine when the surface is planar'",
        ),
        sigma_ref=Ref(Evidence.C, "±2% 관리폭 가정", "σ = 0.03"),
        drift_per_day=0.003,  # pad glazing → 마찰 증가 → 전류 상승
    ),
    "conditioner_force": SensorSpec(
        name="conditioner_force",
        baseline=18.0,  # 4.0 lbs
        sigma=sigma_from_tolerance(18.0, 0.05),  # 0.1125
        unit="N",
        ref=Ref(
            Evidence.B,
            "US9777192 / JEES",
            "pad conditioning: in situ, 4.0 lbs (18 N). glazing 방지용 diamond conditioner",
        ),
        sigma_ref=Ref(Evidence.C, "±5% 가정", "σ ≈ 0.11"),
    ),
    "pad_life_count": SensorSpec(
        name="pad_life_count",
        baseline=800.0,  # 마지막 교체 이후 처리한 wafer 수
        sigma=0.0,  # 카운터
        unit="count",
        ref=Ref(
            Evidence.B,
            "JEES",
            "패드는 1,000~2,000 wafer 후 교체. glazing으로 제거율이 드리프트",
        ),
        sigma_ref=Ref(Evidence.DESIGN, "카운터", "드리프트/교체 트리거"),
    ),
    "pad_temp": SensorSpec(
        name="pad_temp",
        baseline=45.0,
        sigma=sigma_from_tolerance(45.0, 0.03),  # 0.169
        unit="C",
        ref=Ref(
            Evidence.C,
            "JEES (Preston 방정식 맥락)",
            "MRR = Kp × P × V. 마찰열이 pad 온도를 올림. 구체 baseline은 미확보",
        ),
        sigma_ref=Ref(Evidence.C, "±3% 가정", "σ ≈ 0.17"),
        drift_per_day=0.002,
    ),
}


# ------------------------------------------------------------------------------
# Photolithography — Donut의 원인 공정
# ------------------------------------------------------------------------------
# ESWA(2023): "Donut is generally formed due to the redeposition of dissolved
#              photoresist back to the wafer surface"
#   → develop 중 녹은 resist가 wafer 표면에 다시 붙는 것. 중심에서 rinse 액이 퍼지는
#     구조상 도넛 모양이 된다.
#
# bake_temp 110°C는 실제 클린룸 레시피 두 건(AZ 9260, AZ 5214E)과 정확히 일치.

PROCESS_BASELINES["Photolithography"] = {
    "spin_speed": SensorSpec(
        name="spin_speed",
        baseline=3000.0,
        sigma=sigma_from_tolerance(3000.0, 0.01),  # ±1% → σ = 3.75
        unit="rpm",
        ref=Ref(
            Evidence.B,
            "US11809077 (DuPont) / US10497680",
            "'typically spun at up to 4,000 rpm, for example from 200 to 3,000 rpm'",
        ),
        sigma_ref=Ref(Evidence.C, "±1% 관행 가정 (회전 제어)", "σ = 3.75"),
    ),
    "bake_temp": SensorSpec(
        name="bake_temp",
        baseline=110.0,
        sigma=sigma_from_tolerance(110.0, 0.009),  # hotplate ±1°C → σ ≈ 0.25
        unit="C",
        ref=Ref(
            Evidence.B,
            "arXiv:1801.00851 부록 (실제 클린룸 레시피) / US9583344",
            "AZ 9260: 'Soft-bake at 110°C for 165s'. AZ 5214E: '110°C for 90s'. "
            "특허들도 'typical softbakes 90~150°C'",
        ),
        sigma_ref=Ref(Evidence.C, "hotplate 정밀도 ±1°C", "σ ≈ 0.25"),
        drift_per_day=0.001,
    ),
    "exposure_dose": SensorSpec(
        name="exposure_dose",
        baseline=30.0,
        sigma=sigma_from_tolerance(30.0, 0.02),  # ±2% → σ = 0.075
        unit="mJ/cm2",
        ref=Ref(
            Evidence.DESIGN,
            "설계값",
            "실제 레시피들은 6.2 mW/cm² × 노광시간으로 다양. 특정 근거 미확보",
        ),
        sigma_ref=Ref(Evidence.DESIGN, "설계값", "±2% 가정"),
        drift_per_day=0.002,  # 램프 열화
    ),
    "developer_flow": SensorSpec(
        name="developer_flow",
        baseline=200.0,
        sigma=sigma_from_tolerance(200.0, 0.05),  # 1.25
        unit="ml/min",
        ref=Ref(Evidence.DESIGN, "설계값", "구체 근거 미확보"),
        sigma_ref=Ref(Evidence.DESIGN, "설계값", "±5% 가정"),
    ),
    # Donut의 직접 시그니처 — ESWA의 "redeposition" 서술과 직결
    "rinse_flow": SensorSpec(
        name="rinse_flow",
        baseline=500.0,
        sigma=sigma_from_tolerance(500.0, 0.05),  # 3.125
        unit="ml/min",
        ref=Ref(
            Evidence.C,
            "ESWA(2023) 서술에서 유도",
            "'redeposition of dissolved photoresist' → rinse가 부족하면 녹은 resist가 "
            "다시 붙는다. 중심에서 퍼지는 구조상 도넛 모양",
        ),
        sigma_ref=Ref(Evidence.DESIGN, "설계값", "±5% 가정"),
    ),
}


# ------------------------------------------------------------------------------
# Cleaning — Random의 원인 공정
# ------------------------------------------------------------------------------
# Cheon et al.(2019) via Frontiers: Random은 공기 중 입자(airborne particles) 등
# 환경 요인. 특정 장비로 귀속되지 않는다는 것이 정의상의 특징.
#   → 그래서 equipment_correlation이 가장 낮다(0.40).
#
# ⚠️ 이 모듈은 근거가 가장 약하다. 세부 수치는 대부분 [DESIGN].

PROCESS_BASELINES["Cleaning"] = {
    "particle_count": SensorSpec(
        name="particle_count",
        baseline=5.0,
        sigma=float(np.sqrt(5.0)),  # 포아송: std = √mean ≈ 2.24
        unit="count",
        ref=Ref(
            Evidence.B,
            "클린룸 표준 계측 + 포아송 분포",
            "particle counter는 클린룸의 표준 계측기. 카운트 데이터는 std=√mean",
        ),
        sigma_ref=Ref(Evidence.B, "포아송 분포의 성질", "std = sqrt(mean)"),
        drift_per_day=0.01,  # 필터 열화
    ),
    "megasonic_power": SensorSpec(
        name="megasonic_power",
        baseline=300.0,
        sigma=sigma_from_tolerance(300.0, 0.02),  # 1.5
        unit="W",
        ref=Ref(Evidence.DESIGN, "설계값", "megasonic cleaning은 실재하나 구체 수치 미확보"),
        sigma_ref=Ref(Evidence.DESIGN, "설계값", "±2% 가정"),
    ),
    "chemical_conc": SensorSpec(
        name="chemical_conc",
        baseline=2.0,
        sigma=sigma_from_tolerance(2.0, 0.03),  # 0.0075
        unit="%",
        ref=Ref(Evidence.DESIGN, "설계값", "SC-1/SC-2 세정액 농도. 구체 근거 미확보"),
        sigma_ref=Ref(Evidence.DESIGN, "설계값", "±3% 가정"),
        drift_per_day=0.006,  # 약액 소모
    ),
    "dhf_temp": SensorSpec(
        name="dhf_temp",
        baseline=25.0,
        sigma=sigma_from_tolerance(25.0, 0.02),  # 0.0625
        unit="C",
        ref=Ref(Evidence.DESIGN, "설계값", "상온 세정 가정"),
        sigma_ref=Ref(Evidence.DESIGN, "설계값", "±2% 가정"),
    ),
}


# ==============================================================================
# 6. defect → 공정 모듈 → 센서 시그니처 매핑
# ==============================================================================
#
# ## 근거 사슬의 4단계와 각 단계의 근거 수준
#
#   [1] defect → 원인 공정 모듈   ← 문헌이 보증 ([B]) — ESWA(2023), Frontiers(2023)
#   [2] 모듈 → 감시할 센서 선택    ← Etch/CMP만 문헌. 나머지는 우리 설계
#   [3] 이상 시 이동 방향          ← Etch/CMP만 문헌. 나머지는 공정 물리 추론
#   [4] 이동 크기                  ← σ-tier DoE ([C+]) + Goodlin 앵커
#
# ⚠️ **문헌은 [1]까지만 보증한다.** "Center의 원인은 thin film deposition 실패"는
#    논문이 말하지만, "그러면 어느 센서가 어느 방향으로 움직이는가"는 대부분 우리 설계다.
#    이를 숨기지 않고 규칙마다 Evidence 태그로 명시한다.
#
#    예외 (전 사슬 근거 확보):
#      - Edge-Ring: Goodlin(2003)이 Lam 9600에서 pressure/valve/RF를 감시하고
#                   +3 mTorr fault를 실제 주입 → [2][3][4] 전부 [A]
#      - Scratch:   US6428387이 마찰→모터전류 원리를 명시 → [2][3] [B+]
#
# ## equipment_correlation — 시나리오 설계변수 [SCENARIO]
#
# "Edge-Ring wafer의 80%가 문제 장비를 거쳤다"는 물리 상수가 아니라 **우리가 심는 정답지**다.
# 실측 근거가 존재할 수 없고, 찾을 필요도 없다.
#
#   (a) 순서는 문헌 논리로 정당화:
#       장비 고장(Near-full) > 특정 공정 이상(Edge-Ring) > 장비+사람(Scratch)
#       > 간헐적 진동(Loc) > 환경 요인(Random — 정의상 특정 장비로 귀속 안 됨)
#   (b) 절대값은 민감도 분석 대상: 0.9/0.7/0.5/0.3으로 바꿔가며
#       commonality analysis가 원인 장비를 top-1으로 지목하는 성공률을 측정.
#       → "우리 파이프라인은 상관 0.6 이상이면 원인 장비를 정확히 지목한다"가 산출물.

SENSITIVITY_SWEEP = (0.9, 0.7, 0.5, 0.3)  # equipment_correlation 민감도 분석 값


@dataclass(frozen=True)
class DefectRule:
    """defect 하나에 대한 전체 규칙."""

    defect: str
    module: str  # 원인 공정 모듈
    module_ref: Ref  # [1] 모듈 매핑의 근거 (문헌)
    signature_ref: Ref  # [2][3] 센서 시그니처의 근거 (대개 우리 설계)
    tier: DetectionTier  # [4] 탐지 난이도
    deviations: list[Deviation]
    equipment_correlation: float  # [SCENARIO] 민감도 분석 대상
    correlation_rationale: str


DEFECT_RULES: dict[str, DefectRule] = {}


# ------------------------------------------------------------------------------
# Edge-Ring — 근거 최상급 (전 사슬 확보)
# ------------------------------------------------------------------------------
DEFECT_RULES["Edge-Ring"] = DefectRule(
    defect="Edge-Ring",
    module="Etch",
    module_ref=Ref(
        Evidence.B,
        "ESWA(2023) / Frontiers(2023)",
        "'Edge-ring is primarily caused by abnormal etching operations' / "
        "'a ring is due to problems in the etching step'",
    ),
    signature_ref=Ref(
        Evidence.A,
        "Goodlin et al.(2003) — MIT PDF 전문 정독",
        "Lam 9600 etcher의 19개 tool-state 변수(Table 1)에 chamber pressure, RF power, "
        "vat valve position 포함. Table 2에서 pressure fault +3/+2/+1/-2 mTorr를 실제 주입",
    ),
    tier=DetectionTier.STANDARD,  # 2~3σ
    deviations=[
        # Goodlin의 +3 mTorr를 σ 환산: 3 / 1.25 = 2.4σ
        Deviation(
            sensor="chamber_pressure",
            mean_shift_sigma=2.4,
            std_multiplier=1.5,
            direction="up",
            ref=Ref(
                Evidence.A,
                "Goodlin(2003) Table 2, Figure 6",
                "'+3 mtorr deviation in pressure'의 T² contribution plot. σ=1.25 환산 시 2.4σ",
            ),
        ),
        # 압력을 잡으려 밸브가 반응 (물리적 인과)
        Deviation(
            sensor="vat_valve_position",
            mean_shift_sigma=-2.0,
            std_multiplier=1.8,
            direction="down",
            ref=Ref(
                Evidence.B,
                "Goodlin(2003) Table 1 + 물리",
                "vat valve가 감시 변수. 압력 상승 시 밸브가 닫히는 방향으로 반응",
            ),
        ),
        Deviation(
            sensor="rf_power_top",
            mean_shift_sigma=1.5,
            std_multiplier=1.3,
            direction="up",
            ref=Ref(
                Evidence.B,
                "Goodlin(2003) Table 2",
                "RF fault도 주입 대상(+12/+10/+8/-12 W). σ=12.5 환산 시 ~1σ",
            ),
        ),
        # 가장자리 냉각 실패 → 가장자리 과식각 (Edge-Ring의 물리)
        Deviation(
            sensor="backside_he_edge",
            mean_shift_sigma=-2.5,
            std_multiplier=1.4,
            direction="down",
            ref=Ref(
                Evidence.C,
                "US8501499 + 공정 물리 추론",
                "특허가 center/edge He를 따로 제어 → edge 냉각이 실패하면 가장자리 과열·과식각. "
                "다만 'edge He 저하 → Edge-Ring'의 직접 문헌은 미확보",
            ),
        ),
    ],
    equipment_correlation=0.80,
    correlation_rationale=(
        "특정 공정(etch)의 이상이므로 해당 장비에 강하게 귀속. "
        "다만 etch 외 다른 원인도 가능하므로 1.0은 아님"
    ),
)


# ------------------------------------------------------------------------------
# Scratch — motor_current 시그니처가 실무 표준
# ------------------------------------------------------------------------------
DEFECT_RULES["Scratch"] = DefectRule(
    defect="Scratch",
    module="CMP",
    module_ref=Ref(
        Evidence.B,
        "Wang & Bensmial(2006) via ESWA/Frontiers",
        "'a scratch is a result of machine handling problems'",
    ),
    signature_ref=Ref(
        Evidence.B,
        "US6428387 / JEES CMP 기술가이드",
        "'the friction between the pad and the wafer carrier increases... more current is "
        "supplied to the motor... This increase in wafer carrier motor current can be "
        "monitored'. JEES: 'harder pads can increase micro-scratch defects'",
    ),
    tier=DetectionTier.STANDARD,  # 2~3σ
    deviations=[
        # ⭐ 실무 표준 감시 기법 — 마찰 증가가 전류 스파이크로 나타남
        Deviation(
            sensor="motor_current",
            mean_shift_sigma=3.0,
            std_multiplier=2.0,
            direction="spike",
            ref=Ref(
                Evidence.B,
                "US6428387",
                "마찰 변화 → 모터 전류 변화. pad 이상/입자 혼입 시 마찰 급증",
            ),
        ),
        Deviation(
            sensor="polish_pressure",
            mean_shift_sigma=2.5,
            std_multiplier=1.6,
            direction="up",
            ref=Ref(
                Evidence.C,
                "JEES Preston 방정식 + 물리 추론",
                "MRR = Kp × P × V. 압력 과다는 기계적 손상 위험. 직접 문헌은 미확보",
            ),
        ),
        # pad glazing → 마찰열 → 온도 상승 (JEES의 glazing 서술)
        Deviation(
            sensor="pad_temp",
            mean_shift_sigma=2.0,
            std_multiplier=1.5,
            direction="up",
            ref=Ref(
                Evidence.C,
                "JEES glazing 서술에서 유도",
                "패드가 glazing되면 마찰이 증가하고 제거율이 드리프트",
            ),
        ),
        # 슬러리 부족 → 윤활 실패 → 직접 접촉 마모
        Deviation(
            sensor="slurry_flow",
            mean_shift_sigma=-2.0,
            std_multiplier=1.4,
            direction="down",
            ref=Ref(
                Evidence.C,
                "공정 물리 추론",
                "슬러리는 연마입자 공급 + 윤활 기능. 부족하면 pad-wafer 직접 접촉",
            ),
        ),
        Deviation(
            sensor="conditioner_force",
            mean_shift_sigma=-1.5,
            std_multiplier=1.3,
            direction="down",
            ref=Ref(
                Evidence.C,
                "JEES",
                "conditioner가 약하면 glazing이 진행됨",
            ),
        ),
    ],
    equipment_correlation=0.70,
    correlation_rationale=(
        "특정 CMP 장비의 pad/슬러리 문제이나, wafer handling(로봇, 카세트) 등 "
        "장비 외 요인도 섞임 → Edge-Ring보다 낮음"
    ),
)


# ------------------------------------------------------------------------------
# Edge-Loc — RTD(Radial Thermal Delta)의 직접 측정
# ------------------------------------------------------------------------------
DEFECT_RULES["Edge-Loc"] = DefectRule(
    defect="Edge-Loc",
    module="Diffusion",
    module_ref=Ref(
        Evidence.B,
        "Hansen, Nair & Friedman(1997) via ESWA / US7256370",
        "'Edge-Loc is caused by uneven heating during diffusion'. US7256370이 이 문제에 "
        "'RTD (Radial Thermal Delta)'라는 전용 용어를 부여 — 업계에 널리 인지된 고질 문제",
    ),
    signature_ref=Ref(
        Evidence.B,
        "US7256370",
        "'non-uniform heating of substrates from a substrate center to an outer edge'",
    ),
    tier=DetectionTier.BORDERLINE,  # 1~2σ — 1차가 놓치기 시작하는 영역
    deviations=[
        # ⭐ RTD의 직접 시그니처: center는 유지, edge만 하락 → radial delta 발생
        Deviation(
            sensor="wafer_edge_temp",
            mean_shift_sigma=-1.8,
            std_multiplier=1.4,
            direction="down",
            ref=Ref(
                Evidence.B,
                "US7256370",
                "가장자리 온도가 중심보다 낮아지는 것이 RTD. Edge-Loc의 직접 원인",
            ),
        ),
        Deviation(
            sensor="wafer_center_temp",
            mean_shift_sigma=0.3,  # 거의 정상 유지 → delta가 벌어짐
            std_multiplier=1.1,
            direction="any",
            ref=Ref(
                Evidence.B,
                "US7256370",
                "중심은 유지되고 가장자리만 벗어나는 것이 radial delta의 정의",
            ),
        ),
        Deviation(
            sensor="heater_zone_bottom",
            mean_shift_sigma=-1.5,
            std_multiplier=1.3,
            direction="down",
            ref=Ref(
                Evidence.C,
                "물리 추론",
                "heater zone 불균일이 wafer 가장자리 온도 저하로 이어진다는 추론",
            ),
        ),
        Deviation(
            sensor="ramp_rate",
            mean_shift_sigma=1.5,
            std_multiplier=1.5,
            direction="up",
            ref=Ref(
                Evidence.C,
                "US5635414에서 유도",
                "'any substantially greater rate of rise of temperature will damage the "
                "devices' — 급격한 ramp가 열 불균일을 악화",
            ),
        ),
    ],
    equipment_correlation=0.75,
    correlation_rationale="특정 furnace의 heater 불균일 → 해당 장비에 귀속",
)


# ------------------------------------------------------------------------------
# Center — ThinFilm의 반경 방향 균일성 붕괴
# ------------------------------------------------------------------------------
DEFECT_RULES["Center"] = DefectRule(
    defect="Center",
    module="ThinFilm",
    module_ref=Ref(
        Evidence.B,
        "Yuan, Kuo & Bae(2021) via ESWA(2023)",
        "'Center is caused by the failure of the thin film deposition step'",
    ),
    signature_ref=Ref(
        Evidence.B,
        "US4892753 / US5000113 (Applied Materials TEOS CVD)",
        "'uniform susceptor and wafer temperatures, including both absolute temperature "
        "uniformity and spatial uniformity across the susceptor/wafer; uniform gas flow "
        "distribution across the wafer'. 'uniform radial pumping provides uniform gas flow' "
        "→ 반경 방향 균일성이 장비 설계의 핵심 과제이며, 깨지면 중심부에 결함",
    ),
    tier=DetectionTier.STANDARD,  # 2~3σ
    deviations=[
        # ⭐ Diffusion의 RTD와 대칭 구조 — 이번엔 center가 무너진다
        Deviation(
            sensor="susceptor_temp_center",
            mean_shift_sigma=-2.5,
            std_multiplier=1.5,
            direction="down",
            ref=Ref(
                Evidence.B,
                "US4892753",
                "'spatial uniformity across the susceptor/wafer'가 설계 과제 → 실패 시 중심 결함",
            ),
        ),
        Deviation(
            sensor="susceptor_temp_edge",
            mean_shift_sigma=0.2,  # edge는 유지 → thermal delta 발생
            std_multiplier=1.1,
            direction="any",
            ref=Ref(Evidence.B, "US4892753", "center만 무너지고 edge는 유지 → delta 발생"),
        ),
        # showerhead 중심부 막힘 → 전구체 공급 부족
        Deviation(
            sensor="precursor_flow_sih4",
            mean_shift_sigma=-2.2,
            std_multiplier=1.4,
            direction="down",
            ref=Ref(
                Evidence.B,
                "US5000113",
                "'uniform gas flow distribution across the wafer'가 설계 과제. "
                "showerhead 중심 막힘 시 중심부 전구체 부족",
            ),
        ),
        Deviation(
            sensor="deposition_rate",
            mean_shift_sigma=-2.8,
            std_multiplier=1.6,
            direction="down",
            ref=Ref(Evidence.C, "위 두 원인의 결과", "온도·전구체 부족 → 증착 속도 저하"),
        ),
        Deviation(
            sensor="rf_power_hf",
            mean_shift_sigma=-1.2,
            std_multiplier=1.3,
            direction="down",
            ref=Ref(Evidence.C, "물리 추론", "플라즈마 약화 시 증착 균일성 저하"),
        ),
    ],
    equipment_correlation=0.85,
    correlation_rationale=(
        "showerhead/susceptor는 장비 고유의 하드웨어 → 해당 장비에 강하게 귀속. "
        "Edge-Ring(0.80)보다 높은 이유는 하드웨어 결함 성격이 더 강하기 때문"
    ),
)


# ------------------------------------------------------------------------------
# Donut — Photolithography develop 단계
# ------------------------------------------------------------------------------
DEFECT_RULES["Donut"] = DefectRule(
    defect="Donut",
    module="Photolithography",
    module_ref=Ref(
        Evidence.B,
        "ESWA(2023)",
        "'Donut is generally formed due to the redeposition of dissolved photoresist "
        "back to the wafer surface'",
    ),
    signature_ref=Ref(
        Evidence.C,
        "ESWA 서술에서 유도 (센서 선택은 우리 설계)",
        "rinse가 부족하면 녹은 resist가 재증착. 중심에서 액이 퍼지는 구조상 도넛 모양. "
        "다만 어느 센서가 어떻게 움직이는지의 직접 문헌은 미확보",
    ),
    tier=DetectionTier.BORDERLINE,  # 1~2σ
    deviations=[
        Deviation(
            sensor="rinse_flow",
            mean_shift_sigma=-1.8,
            std_multiplier=1.4,
            direction="down",
            ref=Ref(Evidence.C, "ESWA redeposition 서술", "rinse 부족 → resist 재증착"),
        ),
        Deviation(
            sensor="developer_flow",
            mean_shift_sigma=-1.5,
            std_multiplier=1.5,
            direction="down",
            ref=Ref(Evidence.DESIGN, "설계", "현상액 부족 → 불완전 현상"),
        ),
        Deviation(
            sensor="spin_speed",
            mean_shift_sigma=-1.6,
            std_multiplier=1.3,
            direction="down",
            ref=Ref(
                Evidence.DESIGN,
                "설계",
                "회전이 느리면 rinse 액이 원심력으로 배출되지 않아 재증착 가능",
            ),
        ),
    ],
    equipment_correlation=0.70,
    correlation_rationale="track(coater/developer) 장비의 rinse 노즐 문제 → 해당 장비 귀속",
)


# ------------------------------------------------------------------------------
# Loc — vibration_level로 인과 직결 (⭐ 이번 개정의 핵심)
# ------------------------------------------------------------------------------
DEFECT_RULES["Loc"] = DefectRule(
    defect="Loc",
    module="Etch",
    module_ref=Ref(
        Evidence.B,
        "Hansen & Thyregod(1998) via ESWA(2023)",
        "'Local is commonly caused by excessive vibration of a specific machine'",
    ),
    signature_ref=Ref(
        Evidence.C,
        "vibration_level 센서 신설로 인과 직결",
        "기존에는 진동 센서가 없어 pressure/RF의 std 증가로 간접 표현했으나, "
        "'진동 → 챔버 압력 변동성 증가'는 미확인 추론이었다. 진동 센서를 두면 "
        "'진동 원인 → 진동 센서 이상'으로 인과가 직결된다",
    ),
    # ⭐ HIDDEN tier — 2계층 FDC 설계의 존재 이유를 검증하는 케이스
    #    진동은 평균을 유지한 채 변동성만 키운다 → 3σ chart는 평균 이동을 보므로 무력
    #    → 2차 탐지기(Autoencoder)만 잡을 수 있다
    tier=DetectionTier.HIDDEN,
    deviations=[
        # ⭐ 인과 직결: 진동이 원인 → 진동 센서가 직접 반응
        Deviation(
            sensor="vibration_level",
            mean_shift_sigma=0.5,  # 평균은 거의 안 움직임
            std_multiplier=3.0,  # 변동성만 3배 → 1차 무력
            direction="unstable",
            ref=Ref(
                Evidence.C,
                "Hansen & Thyregod 'excessive vibration' + 물리",
                "진동은 평균을 유지한 채 변동성만 키운다. mean_shift ≈ 0, std 3배가 "
                "물리적으로 정확한 시그니처",
            ),
        ),
        # 진동이 다른 센서로 전파 (간접 효과)
        Deviation(
            sensor="chamber_pressure",
            mean_shift_sigma=0.0,
            std_multiplier=2.5,
            direction="unstable",
            ref=Ref(
                Evidence.DESIGN,
                "설계 (미확인 추론)",
                "진동이 챔버 압력 변동성을 키운다는 것은 물리적으로 그럴듯하나 문헌 미확보",
            ),
        ),
        Deviation(
            sensor="rf_power_top",
            mean_shift_sigma=0.0,
            std_multiplier=2.2,
            direction="unstable",
            ref=Ref(Evidence.DESIGN, "설계", "진동 → RF 매칭 불안정 (추론)"),
        ),
        Deviation(
            sensor="vat_valve_position",
            mean_shift_sigma=0.0,
            std_multiplier=2.0,
            direction="unstable",
            ref=Ref(Evidence.DESIGN, "설계", "압력 변동에 밸브가 계속 반응 (추론)"),
        ),
    ],
    equipment_correlation=0.60,
    correlation_rationale=(
        "특정 장비의 진동이지만 간헐적이고, 진동은 인접 장비로도 전파 가능 → 중간 수준"
    ),
)


# ------------------------------------------------------------------------------
# Random — 환경 요인 (정의상 장비 귀속이 약함)
# ------------------------------------------------------------------------------
DEFECT_RULES["Random"] = DefectRule(
    defect="Random",
    module="Cleaning",
    module_ref=Ref(
        Evidence.B,
        "Cheon et al.(2019) via Frontiers(2023)",
        "Random은 공기 중 입자(airborne particles) 등 환경 요인",
    ),
    signature_ref=Ref(
        Evidence.B,
        "클린룸 표준 계측",
        "particle counter는 클린룸의 표준 계측기. 입자 증가 = 카운트 증가는 자명",
    ),
    tier=DetectionTier.STANDARD,
    deviations=[
        Deviation(
            sensor="particle_count",
            mean_shift_sigma=3.0,
            std_multiplier=2.0,
            direction="up",
            ref=Ref(Evidence.B, "자명", "입자 오염 → 카운트 증가"),
        ),
        Deviation(
            sensor="chemical_conc",
            mean_shift_sigma=-2.0,
            std_multiplier=1.5,
            direction="down",
            ref=Ref(Evidence.DESIGN, "설계", "세정액 농도 저하 → 세정력 부족 (추론)"),
        ),
        Deviation(
            sensor="megasonic_power",
            mean_shift_sigma=-2.0,
            std_multiplier=1.4,
            direction="down",
            ref=Ref(Evidence.DESIGN, "설계", "megasonic 출력 저하 → 입자 제거 실패 (추론)"),
        ),
    ],
    # ⭐ 가장 낮음 — 환경 요인은 정의상 특정 장비로 귀속되지 않는다
    equipment_correlation=0.40,
    correlation_rationale=(
        "공기 중 입자 등 환경 요인은 정의상 특정 장비로 귀속되지 않는다. "
        "commonality analysis가 원인 장비를 찾기 가장 어려운 케이스 — "
        "이 낮은 상관이 파이프라인의 탐지 한계를 시험한다"
    ),
)


# ------------------------------------------------------------------------------
# Near-full — 장비 고장 (모든 센서 대폭 이탈)
# ------------------------------------------------------------------------------
DEFECT_RULES["Near-full"] = DefectRule(
    defect="Near-full",
    module="Etch",
    module_ref=Ref(
        Evidence.B,
        "ESWA(2023) 'receipt failure'",
        "⚠️ 단, 장비 고장은 어느 모듈에서든 가능하다. Etch 배정은 시나리오상의 설정",
    ),
    signature_ref=Ref(
        Evidence.C_PLUS,
        "설계 (탐지 난이도 OBVIOUS 배치용)",
        "장비 고장이므로 여러 센서가 동시에 크게 이탈하는 것이 자연스럽다. "
        "1차 방어선(3σ chart)이 즉시 탐지해야 하는 케이스로 배치",
    ),
    tier=DetectionTier.OBVIOUS,  # 4~6σ — 1차가 즉시 잡아야 함
    deviations=[
        Deviation(
            sensor="chamber_pressure",
            mean_shift_sigma=5.0,
            std_multiplier=3.0,
            direction="up",
            ref=Ref(Evidence.C_PLUS, "OBVIOUS tier 설계", "5σ — 명백한 이상"),
        ),
        Deviation(
            sensor="rf_power_top",
            mean_shift_sigma=-5.0,
            std_multiplier=3.0,
            direction="down",
            ref=Ref(Evidence.C_PLUS, "OBVIOUS tier 설계", "RF 실패"),
        ),
        Deviation(
            sensor="gas_flow_cf4",
            mean_shift_sigma=-4.5,
            std_multiplier=2.5,
            direction="down",
            ref=Ref(Evidence.C_PLUS, "OBVIOUS tier 설계", "가스 공급 실패"),
        ),
        Deviation(
            sensor="backside_he_center",
            mean_shift_sigma=-5.0,
            std_multiplier=2.5,
            direction="down",
            ref=Ref(Evidence.C_PLUS, "OBVIOUS tier 설계", "냉각 실패"),
        ),
        Deviation(
            sensor="vat_valve_position",
            mean_shift_sigma=4.0,
            std_multiplier=2.5,
            direction="up",
            ref=Ref(Evidence.C_PLUS, "OBVIOUS tier 설계", "밸브 이상"),
        ),
    ],
    # 가장 높음 — 장비 고장은 해당 장비에 강하게 귀속
    equipment_correlation=0.95,
    correlation_rationale="장비 고장(equipment failure)은 해당 장비에 강하게 귀속",
)


# 'none'은 규칙 없음 — 모든 센서가 baseline 거동
NORMAL_LABEL = "none"

WM811K_LABELS = [
    NORMAL_LABEL,
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


# ==============================================================================
# 7. 장비(tool) 계층
# ==============================================================================
#
# ## 왜 장비 개념이 필수인가
#
# 센서는 **장비에 달려 있다.** "Etch 공정의 압력"이라는 건 없고 "ETCH-07의 압력"만 있다.
# 그리고 lot마다 어느 장비를 거쳤는지가 다르다:
#
#     LOT-4231: PHOTO-03 → ETCH-07 → TF-02 → CMP-05 → CLEAN-01
#     LOT-4232: PHOTO-01 → ETCH-02 → TF-02 → CMP-05 → CLEAN-04
#
# ETCH-07이 고장 나면 그 장비를 거친 lot에서만 Edge-Ring이 나온다. 불량 lot들의 이력을
# 모아 "공통으로 거친 장비"를 찾으면 ETCH-07이 지목된다 — **이것이 commonality analysis**.
#
# **장비 ID가 없으면 센서 이상과 wafer 불량을 이을 방법이 원천적으로 없다.**
# 4개 테이블(sensor_traces, fdc_alarms, wafer_results, lot_history)의 join 키가 장비 ID다.

# 모듈당 장비 대수. commonality analysis가 의미 있으려면 모듈당 여러 대가 필요하다
# (1대뿐이면 모든 lot이 그 장비를 거치므로 원인 특정이 불가능).
TOOLS_PER_MODULE: dict[str, int] = {
    "Photolithography": 8,
    "Etch": 10,
    "ThinFilm": 8,
    "Diffusion": 6,
    "CMP": 8,
    "Cleaning": 10,
}  # 합계 50대

TOOL_ID_PREFIX: dict[str, str] = {
    "Photolithography": "PHOTO",
    "Etch": "ETCH",
    "ThinFilm": "TF",
    "Diffusion": "DIFF",
    "CMP": "CMP",
    "Cleaning": "CLEAN",
}

# 장비 가동률 — 장비는 항상 신호를 내지 않는다. wafer가 챔버 안에 있을 때
# trace 데이터가 의미 있고, idle에서는 발행이 줄거나 멈춘다. fab 통상 70~85%.
TOOL_UTILIZATION = 0.8

# 샘플링 레이트 (Hz)
#   US11029673: "The recent norm for data collection rates on semiconductor process tools
#                is 1 Hz... ITRS predicts 100 Hz in three years... Most experts believe
#                a more realistic rate will be 10 Hz"
SAMPLING_RATE_BASELINE = 1.0  # 업계 현 표준
SAMPLING_RATE_TARGET = 10.0  # 전문가 현실 전망


@dataclass(frozen=True)
class Tool:
    """장비 1대."""

    tool_id: str  # "ETCH-07"
    module: str  # "Etch"
    core_sensors: dict[str, SensorSpec]  # 핵심 센서 (defect 인과 연결)
    background_sensors: dict[str, SensorSpec]  # 배경 채널 (~240개)

    @property
    def all_sensors(self) -> dict[str, SensorSpec]:
        return {**self.core_sensors, **self.background_sensors}

    @property
    def channel_count(self) -> int:
        return len(self.all_sensors)


def _tool_seed(tool_id: str, base: int = 42) -> int:
    """장비 ID에서 결정론적 시드를 만든다.

    hash()는 프로세스마다 값이 달라지므로(PYTHONHASHSEED) 쓸 수 없다.
    같은 tool_id는 언제나 같은 채널·같은 mean/std를 가져야 한다 — 실제 장비가 그렇고,
    그래야 baseline 통계를 학습(AE)하고 3σ 관리한계를 계산할 수 있다.
    """
    digest = hashlib.sha256(tool_id.encode()).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**31)


def build_background_channels(tool_id: str, module: str) -> dict[str, SensorSpec]:
    """장비별 배경 채널을 생성한다. 같은 tool_id면 항상 같은 결과.

    장비마다 mean이 조금씩 다른 것도 현실적이다 — ETCH-07과 ETCH-08은 같은 모델이어도
    개체차가 있다. 이것이 실무에서 **장비별로 관리한계를 따로 잡는 이유**이고,
    MFC 문헌이 말하는 "chamber matching"이 필요한 이유다.
    """
    rng = np.random.default_rng(_tool_seed(tool_id))
    mix = BACKGROUND_CHANNEL_MIX[module]

    channels: dict[str, SensorSpec] = {}
    for archetype_name, count in mix.items():
        archetype = CHANNEL_ARCHETYPES[archetype_name]
        for i in range(count):
            spec = archetype.sample_spec(rng, i)
            channels[spec.name] = spec

    return channels


# 핵심 센서 중 HELD(에피소드 내내 평탄)인 것들. 나머지는 RECIPE(사다리꼴)로 본다.
#   - 상시 유지되는 열/기계 setpoint: 존/서셉터/챔버벽/베이크 온도, 전극 간격
#   - 배스 유지값(습식): 화학농도, DHF 온도
#   - 장비 속성 자체: vibration_level (웨이퍼 유무와 무관)
#   - 카운터(sigma=0)는 정의상 평탄이므로 HELD로 함께 둔다
# 등급: 개별 분류는 [C]/[DESIGN] — 물리적 성격에 근거하되 경계 사례는 관행 판단.
HELD_CORE_SENSORS: frozenset[str] = frozenset({
    "bake_temp",
    "chamber_wall_temp",
    "heater_zone_top",
    "heater_zone_center",
    "heater_zone_bottom",
    "susceptor_temp_center",
    "susceptor_temp_edge",
    "electrode_spacing",
    "chemical_conc",
    "dhf_temp",
    "vibration_level",
    "pad_life_count",
    "chamber_clean_cycle_count",
})


def _apply_core_actuation(core: dict[str, SensorSpec]) -> dict[str, SensorSpec]:
    """핵심 센서에 actuation을 부여한다. HELD_CORE_SENSORS면 HELD, 아니면 RECIPE."""
    out: dict[str, SensorSpec] = {}
    for name, spec in core.items():
        act = Actuation.HELD if name in HELD_CORE_SENSORS else Actuation.RECIPE
        out[name] = replace(spec, actuation=act)
    return out


def build_tool(tool_id: str, module: str) -> Tool:
    """장비 1대를 구성한다 (핵심 센서 + 배경 채널)."""
    core = _apply_core_actuation(PROCESS_BASELINES[module])
    n_core = len(core)
    n_background_target = CHANNELS_PER_TOOL - n_core

    background = build_background_channels(tool_id, module)

    # BACKGROUND_CHANNEL_MIX는 모듈당 245개 근처로 설계했으나 핵심 센서 수가
    # 모듈마다 다르므로 총 250에 맞춰 조정한다.
    if len(background) > n_background_target:
        keys = sorted(background)[:n_background_target]
        background = {k: background[k] for k in keys}
    elif len(background) < n_background_target:
        # 부족하면 온도 채널로 채운다 (가장 흔한 채널 종류)
        rng = np.random.default_rng(_tool_seed(tool_id) + 1)
        archetype = CHANNEL_ARCHETYPES[ARCHETYPE_TEMPERATURE]
        idx = 900  # 기존 인덱스와 충돌 방지
        while len(background) < n_background_target:
            spec = archetype.sample_spec(rng, idx)
            background[spec.name] = spec
            idx += 1

    return Tool(
        tool_id=tool_id,
        module=module,
        core_sensors=dict(core),
        background_sensors=background,
    )


def build_fab(modules: dict[str, int] | None = None) -> dict[str, Tool]:
    """fab 전체 장비를 구성한다. 기본 50대."""
    modules = modules or TOOLS_PER_MODULE
    tools: dict[str, Tool] = {}

    for module, n in modules.items():
        prefix = TOOL_ID_PREFIX[module]
        for i in range(1, n + 1):
            tool_id = f"{prefix}-{i:02d}"
            tools[tool_id] = build_tool(tool_id, module)

    return tools


def expected_generation_rate(
    n_tools: int = 50,
    channels_per_tool: int = CHANNELS_PER_TOOL,
    sampling_hz: float = SAMPLING_RATE_BASELINE,
    utilization: float = TOOL_UTILIZATION,
) -> float:
    """생성량(msg/s)을 계산한다.

    생성량 = 장비 수 × 가동률 × 장비당 채널 × 샘플링 레이트

    기준:   50 × 0.8 × 250 × 1 Hz  =  10,000 msg/s   (업계 현 표준 레이트)
    목표:   50 × 0.8 × 250 × 10 Hz = 100,000 msg/s   (전문가 전망 레이트)

    ⚠️ 생성량(우리가 통제하는 입력)과 처리량(파이프라인이 소화하는 출력)은 다르다.
       부하 테스트 = 생성량을 올려가며 처리량 한계(sustainable throughput)를 찾는 것.
    """
    return n_tools * utilization * channels_per_tool * sampling_hz


def expected_false_alarm_rate(
    n_tools: int = 50,
    background_per_tool: int = 240,
    sampling_hz: float = SAMPLING_RATE_BASELINE,
    sigma_limit: float = 3.0,
) -> float:
    """1차 방어선(채널별 3σ chart)의 기준선 오경보율(건/초).

    배경 채널이 정상 변동을 가지므로 3σ chart가 자연스럽게 오경보를 낸다.
    정규분포 3σ 밖 확률 = 0.27%.

        240채널 × 50대 × 0.0027 ≈ 32건/초 (1 Hz 기준)

    ⚠️ **이것은 버그가 아니라 필수 기능이다.**
    오경보가 없으면 2차 탐지기(T²/AE)의 존재 이유를 검증할 수 없다.

    실무 근거:
      - Goodlin(2003): 개별 SPC 차트는 오경보 12건, T² 차트는 1건 (92% 감소)
      - FDC 실무 가이드: "AI-powered FDC reduces false alarm rates by 60–70%...
        Unlike rule-based FDC that triggers on single-sensor thresholds, ML-based FDC
        models multivariate equipment signatures"
      → "1차(단변량 3σ) = 오경보 많음 → 2차(다변량) = 60~70% 감소"가 실무의 표준 서사.
        우리 2계층 FDC 설계와 정확히 일치하며, 이것이 검증 목표가 된다.
    """
    from math import erf, sqrt

    # 양측 3σ 밖 확률
    p_outside = 1.0 - erf(sigma_limit / sqrt(2.0))
    n_channels = n_tools * background_per_tool
    return n_channels * p_outside * sampling_hz


# ==============================================================================
# 8. 검증 유틸 (정합성 자체 점검)
# ==============================================================================


def validate() -> list[str]:
    """매핑 테이블의 정합성을 점검한다. 문제가 있으면 메시지 리스트를 반환."""
    problems: list[str] = []

    # (1) 모든 defect 규칙의 센서가 해당 모듈에 실재하는가
    for name, rule in DEFECT_RULES.items():
        if rule.module not in PROCESS_BASELINES:
            problems.append(f"{name}: 모듈 '{rule.module}'이 PROCESS_BASELINES에 없음")
            continue
        module_sensors = PROCESS_BASELINES[rule.module]
        for dev in rule.deviations:
            if dev.sensor not in module_sensors:
                problems.append(
                    f"{name}: 센서 '{dev.sensor}'가 {rule.module} 모듈에 없음"
                )

    # (2) σ-tier와 실제 mean_shift_sigma가 일치하는가
    tier_ranges = {
        DetectionTier.OBVIOUS: (4.0, 6.5),
        DetectionTier.STANDARD: (2.0, 3.5),
        DetectionTier.BORDERLINE: (1.0, 2.0),
        DetectionTier.HIDDEN: (0.0, 0.6),
    }
    for name, rule in DEFECT_RULES.items():
        lo, hi = tier_ranges[rule.tier]
        max_shift = max(abs(d.mean_shift_sigma) for d in rule.deviations)
        if not (lo <= max_shift <= hi):
            problems.append(
                f"{name}: tier={rule.tier.value}({lo}~{hi}σ)인데 "
                f"최대 이동이 {max_shift}σ — 불일치"
            )

    # (3) HIDDEN tier는 std_multiplier가 충분히 커야 (2차 탐지기만 잡을 수 있도록)
    for name, rule in DEFECT_RULES.items():
        if rule.tier != DetectionTier.HIDDEN:
            continue
        max_mult = max(d.std_multiplier for d in rule.deviations)
        if max_mult < 2.0:
            problems.append(
                f"{name}: HIDDEN tier인데 std_multiplier 최대가 {max_mult} — "
                "2차 탐지기도 못 잡을 수 있음"
            )

    # (4) equipment_correlation 순서 (문헌 논리)
    expected_order = ["Near-full", "Center", "Edge-Ring", "Edge-Loc", "Scratch", "Donut", "Loc", "Random"]
    corrs = [(d, DEFECT_RULES[d].equipment_correlation) for d in expected_order if d in DEFECT_RULES]
    for (n1, c1), (n2, c2) in zip(corrs, corrs[1:]):
        if c1 < c2:
            problems.append(f"correlation 순서 위반: {n1}({c1}) < {n2}({c2})")

    # (5) CV(변동계수)가 물리적으로 타당한가
    for module, sensors in PROCESS_BASELINES.items():
        for sname, spec in sensors.items():
            if spec.sigma == 0:  # 카운터는 제외
                continue
            if spec.cv > 0.5:
                problems.append(
                    f"{module}.{sname}: CV={spec.cv:.1%} — 너무 큼 (측정 불가능 수준)"
                )

    # (6) HELD_CORE_SENSORS의 이름이 실제 핵심 센서로 존재하는가
    #     오타가 있으면 조용히 RECIPE로 떨어져 트레이스 형태가 잘못된다 (경고 없이).
    all_core = {s for sensors in PROCESS_BASELINES.values() for s in sensors}
    for sname in sorted(HELD_CORE_SENSORS - all_core):
        problems.append(
            f"HELD_CORE_SENSORS의 '{sname}'가 어느 모듈에도 없음 — "
            "오타면 조용히 RECIPE(사다리꼴)로 처리됨"
        )

    # (7) 카운터(sigma=0)는 정의상 평탄 → HELD여야 한다
    for module, sensors in PROCESS_BASELINES.items():
        for sname, spec in sensors.items():
            if spec.sigma == 0 and sname not in HELD_CORE_SENSORS:
                problems.append(
                    f"{module}.{sname}: 카운터(sigma=0)인데 HELD_CORE_SENSORS에 없음 "
                    "— RECIPE로 잡히면 ramp 구간에서 값이 깎인다"
                )

    return problems


def summarize() -> str:
    """테이블 요약 (근거 등급 분포 포함)."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("FabScope defect-sensor 매핑 테이블")
    lines.append("=" * 78)

    # 모듈별 핵심 센서
    lines.append("\n[핵심 센서] (defect 인과 연결, deviation 규칙 적용)")
    total_core = 0
    for module, sensors in PROCESS_BASELINES.items():
        total_core += len(sensors)
        lines.append(f"\n  {module} ({len(sensors)}개)")
        for name, spec in sensors.items():
            cv_str = f"CV={spec.cv:.2%}" if spec.sigma else "counter"
            lines.append(
                f"    {name:<28} {spec.baseline:>9.2f} ± {spec.sigma:<7.3f} "
                f"{spec.unit:<10} {cv_str:<12} [{spec.sigma_ref.grade.value}]"
            )

    # defect 규칙
    lines.append("\n\n[defect 규칙]")
    lines.append(
        f"  {'defect':<12} {'모듈':<18} {'tier':<12} {'corr':<6} {'센서 수':<8} 근거"
    )
    lines.append("  " + "-" * 74)
    for name, rule in DEFECT_RULES.items():
        lines.append(
            f"  {name:<12} {rule.module:<18} {rule.tier.value:<12} "
            f"{rule.equipment_correlation:<6.2f} {len(rule.deviations):<8} "
            f"[{rule.module_ref.grade.value}]→[{rule.signature_ref.grade.value}]"
        )

    # 근거 등급 분포
    lines.append("\n\n[근거 등급 분포] — 시그니처(센서 선택+방향)의 근거")
    grade_count: dict[str, int] = {}
    for rule in DEFECT_RULES.values():
        for dev in rule.deviations:
            g = dev.ref.grade.value if dev.ref else "DESIGN"
            grade_count[g] = grade_count.get(g, 0) + 1
    for g in sorted(grade_count):
        lines.append(f"  [{g}]: {grade_count[g]}개 규칙")
    lines.append(
        "\n  ⚠️ 문헌은 [defect→모듈]까지만 보증한다. 센서 시그니처는 대부분 우리 설계이며,"
    )
    lines.append(
        "     예외는 Edge-Ring(Goodlin 2003)과 Scratch(US6428387)뿐이다."
    )

    # 채널 구조
    lines.append("\n\n[채널 구조]")
    lines.append(f"  장비당 총 채널     : {CHANNELS_PER_TOOL}")
    lines.append(f"  핵심 센서 (모듈별) : {total_core // len(PROCESS_BASELINES)}개 평균")
    lines.append(f"  배경 채널          : 나머지 (~240개)")
    lines.append(f"  장비 수            : {sum(TOOLS_PER_MODULE.values())}대")
    lines.append(
        f"  fab 전체 채널      : {sum(TOOLS_PER_MODULE.values()) * CHANNELS_PER_TOOL:,}개"
    )
    lines.append("    (SECOM 590개는 '라인 전체' 집계라 층위가 다름 — 우리가 21배 많음)")

    # 생성량
    lines.append("\n\n[생성량 / 처리 목표]")
    base = expected_generation_rate(sampling_hz=SAMPLING_RATE_BASELINE)
    target = expected_generation_rate(sampling_hz=SAMPLING_RATE_TARGET)
    lines.append(f"  기준 (1 Hz)  : {base:>9,.0f} msg/s   ← 업계 현 표준")
    lines.append(f"  목표 (10 Hz) : {target:>9,.0f} msg/s   ← 전문가 전망 레이트")
    lines.append(f"  바이트 환산  : {base * 200 / 1e6:.1f} MB/s ≈ {base * 200 * 86400 / 1e9:.0f} GB/일")
    lines.append("    (IIoT 문헌의 '장비당 1~10 GB/일 × 50대 = 50~500 GB/일' 범위 안)")

    # 오경보
    far = expected_false_alarm_rate()
    lines.append(f"\n  1차 3σ chart 기준선 오경보율: {far:.1f} 건/초")
    lines.append("    → 2차 탐지기(T²/AE)로 60~70% 감소가 목표 (FDC 실무 벤치마크)")
    lines.append("    → Goodlin(2003) 실측: 12건 → 1건 (92% 감소)")

    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize())

    print("\n\n" + "=" * 78)
    print("정합성 검증")
    print("=" * 78)
    issues = validate()
    if issues:
        print(f"\n⚠️  문제 {len(issues)}건:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 모든 검증 통과")

    # 장비 구성 확인
    print("\n\n" + "=" * 78)
    print("장비 구성 (샘플)")
    print("=" * 78)
    fab = build_fab()
    print(f"\n총 장비: {len(fab)}대")

    sample = fab["ETCH-07"]
    print(f"\n{sample.tool_id} ({sample.module})")
    print(f"  총 채널      : {sample.channel_count}")
    print(f"  핵심 센서    : {len(sample.core_sensors)}개")
    print(f"  배경 채널    : {len(sample.background_sensors)}개")

    # 장비별 개체차 확인 (같은 모듈, 다른 장비)
    other = fab["ETCH-08"]
    k = sorted(sample.background_sensors)[0]
    s1 = sample.background_sensors[k]
    s2 = other.background_sensors[k]
    print(f"\n장비별 개체차 (같은 채널 '{k}'):")
    print(f"  ETCH-07: mean={s1.baseline:.2f}, std={s1.sigma:.4f}")
    print(f"  ETCH-08: mean={s2.baseline:.2f}, std={s2.sigma:.4f}")
    print("  → 같은 모델도 개체차가 있다. 실무에서 장비별 관리한계를 따로 잡는 이유")

    # 결정론 확인
    fab2 = build_fab()
    s3 = fab2["ETCH-07"].background_sensors[k]
    assert abs(s1.baseline - s3.baseline) < 1e-9, "결정론 위반!"
    print("\n✅ 결정론 확인: 같은 tool_id는 항상 같은 채널 스펙")

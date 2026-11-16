"""Synthetic wetland scenarios for demonstration and validation.

All positions, sources and stations are fictional.  The model is deliberately
simple and documented so an independent checker can recompute arrivals:

* Flat 2-D plane in metres, constant propagation speed (343 m/s by default).
* Each station boot is one affine clock::

      local = boot_start_local + (1 + drift_ppm*1e-6) * (reference - boot_start_ref)

  ``boot_start_local`` is the station-clock counter at power-on and may jump
  arbitrarily between boots.  Drift differs between boots but stays tiny
  (tens of ppm), so a boot needs a piecewise-linear map only if the drift
  changes mid-boot — the default scenarios have one segment per boot.
* Calibration sources sit at known coordinates and emit at known *reference*
  instants.  Every station records each emission it can hear.
* Sound sources are birds/amphibians: a class label, a feature vector, and a
  list of emission instants.

The generator writes the station-side view (local readings, upload batches)
plus a ``truth.json`` that lets a checker recompute arrivals from first
principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

SPEED_MPS = 343.0
REFERENCE_EPOCH = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
LOCAL_EPOCH = datetime(2026, 11, 1)


def us(dt: datetime) -> int:
    return int(round(dt.timestamp() * 1_000_000))


def ref_at(seconds: float) -> datetime:
    return REFERENCE_EPOCH + timedelta(seconds=seconds)


def ref_iso(seconds: float) -> str:
    return ref_at(seconds).isoformat(timespec="microseconds").replace("+00:00", "Z")


def local_iso(local_us: int) -> str:
    dt = LOCAL_EPOCH + timedelta(microseconds=local_us)
    return dt.isoformat(timespec="microseconds")


@dataclass
class Boot:
    """One power-on period of a station."""

    boot_id: str
    boot_start_ref_s: float
    boot_start_local_s: float
    drift_ppm: float
    reboot_reason: str | None = None
    # Extra fixed offset (microseconds) added to every local reading of this
    # boot.  Used by the fault-attribution scenario to emulate a clock offset
    # the maintenance diagnosis must attribute to the clock rather than to
    # geometry or propagation.
    extra_offset_us: float = 0.0

    # Late offset step: the clock jumps by a fixed amount late in the boot
    # when too few calibration anchors remain to open a second piecewise
    # segment, so a same-sign residual survives the per-boot clock fit.
    offset_step_s: float | None = None
    offset_step_us: float = 0.0

    def local_us(self, reference_arrival_us: int) -> int:
        a = 1.0 + self.drift_ppm * 1e-6
        step = (
            self.offset_step_us
            if self.offset_step_s is not None
            and reference_arrival_us >= us(ref_at(self.offset_step_s))
            else 0.0
        )
        return int(
            round(
                self.boot_start_local_s * 1e6
                + a * (reference_arrival_us - us(ref_at(self.boot_start_ref_s)))
                + self.extra_offset_us
                + step
            )
        )

    def truth(self) -> dict:
        a = 1.0 + self.drift_ppm * 1e-6
        b = self.boot_start_local_s * 1e6 - a * us(ref_at(self.boot_start_ref_s))
        return {
            "boot_id": self.boot_id,
            "boot_start_ref": ref_iso(self.boot_start_ref_s),
            "boot_start_local": local_iso(int(self.boot_start_local_s * 1e6)),
            "drift_ppm": self.drift_ppm,
            "slope": a,
            "intercept_us": b,
            "reboot_reason": self.reboot_reason,
            "extra_offset_us": self.extra_offset_us,
            "offset_step_s": self.offset_step_s,
            "offset_step_us": self.offset_step_us,
        }

    def covers(self, reference_s: float) -> bool:
        return reference_s >= self.boot_start_ref_s


@dataclass
class StationSpec:
    id: str
    x: float
    y: float
    radius: float
    boots: list[Boot]
    upload_lag_s: tuple[float, float] = (30.0, 120.0)
    late: bool = False  # uploads only in the late phase
    note: str | None = None
    # The station is *registered* in the service at a shifted measurement
    # point.  Truth arrivals are generated at (x, y); the service believes it
    # sits at (x + registered_dx, y + registered_dy).  This emulates a
    # mis-recorded coordinate that the hypothesis diagnosis must recover.
    registered_dx: float = 0.0
    registered_dy: float = 0.0

    def registered_xy(self) -> tuple[float, float]:
        return self.x + self.registered_dx, self.y + self.registered_dy

    def boot_at(self, reference_s: float) -> Boot:
        chosen = self.boots[0]
        for boot in self.boots:
            if boot.covers(reference_s) and boot.boot_start_ref_s >= chosen.boot_start_ref_s:
                chosen = boot
        return chosen


@dataclass
class SourceSpec:
    id: str
    class_label: str
    x: float
    y: float
    emissions_s: list[float]
    feature: list[float]
    temporal_sigma_ms: float = 0.0


@dataclass
class CalSourceSpec:
    id: str
    x: float
    y: float
    emissions_s: list[float]
    late_from_s: float | None = None  # emissions at/after this instant upload late


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    stations: list[StationSpec]
    sources: list[SourceSpec]
    cal_sources: list[CalSourceSpec]
    area: dict[str, float]
    observation_uncertainty_ms: float = 25.0
    anchor_noise_ms: float = 4.0
    feature_noise: float = 0.015
    closure_margin_s: float = 180.0
    # Actual regional propagation speed used to generate arrivals.  The service
    # keeps 343 m/s as its baseline, so a different value emulates a regional
    # propagation change the hypothesis diagnosis must attribute correctly.
    propagation_speed_mps: float = SPEED_MPS
    # station_id -> extra delay (ms) injected into one anchor reading, so the
    # diagnosis outlier rejection has a documented anomaly to exclude.
    anchor_outliers_ms: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)

    def timeline_end_s(self) -> float:
        sound_end = max((e for s in self.sources for e in s.emissions_s), default=600.0)
        cal_end = max((e for c in self.cal_sources for e in c.emissions_s), default=0.0)
        return max(sound_end, cal_end) + self.closure_margin_s


BOOT_BASE_LOCAL_S = 1_158_000.0  # ~2026-11-14 on the fictional station calendar


def _boots_standard() -> Boot:
    return Boot("b1", -3600.0, BOOT_BASE_LOCAL_S, 35.0)


def build_scenario(name: str) -> Scenario:
    builders = {
        "overlapping": _overlapping,
        "reboot": _reboot,
        "collinear": _collinear,
        "duplicates": _duplicates,
        "revision": _revision,
        "diagnosis": _diagnosis,
        "sparse_anchors": _sparse_anchors,
    }
    if name not in builders:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(builders)}")
    return builders[name]()


def scenario_names() -> list[str]:
    return [
        "overlapping", "reboot", "collinear", "duplicates", "revision",
        "diagnosis", "sparse_anchors",
    ]


# --------------------------------------------------------------------------- #
# 1. Overlapping dual sources
# --------------------------------------------------------------------------- #
def _overlapping() -> Scenario:
    cal = [
        CalSourceSpec("C1", 100.0, -150.0, [-2400.0 + 115.0 * i for i in range(37)]),
        CalSourceSpec("C2", 1650.0, 900.0, [-2280.0 + 110.0 * i for i in range(37)]),
    ]
    stations = [
        StationSpec("S1", 300.0, 100.0, 1700.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1400.0, 150.0, 1700.0, [_boots_standard()], (40.0, 110.0)),
        StationSpec("S3", 1050.0, 950.0, 1700.0, [_boots_standard()], (55.0, 140.0)),
        StationSpec("S4", 550.0, 600.0, 1600.0, [_boots_standard()], (20.0, 60.0)),
    ]
    sources = [
        SourceSpec("A", "chirp", 760.0, 300.0, [120.0, 900.0, 1700.0], [1.0, 0.30, 0.20, 0.90]),
        SourceSpec("B", "chirp", 880.0, 480.0, [123.5, 905.0, 1695.0], [1.0, 0.42, 0.34, 0.86]),
    ]
    return Scenario(
        name="overlapping",
        title="Two chirp sources calling almost simultaneously",
        description=(
            "Two distinct chirp sources 200 m apart call a few seconds apart; S2 and S3 "
            "hear both. One animal heard by four stations must be one event, yet the two "
            "animals must stay two events even though their calls overlap in time. A "
            "station can contribute at most one record to each event."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -300.0, "x_max": 2100.0, "y_min": -500.0, "y_max": 1300.0},
        expected={
            "events": 6,
            "events_with_4_stations": 6,
            "distinct_sources": 2,
            "note": "each source emits 3 calls; every call should be heard by all four stations",
        },
    )


# --------------------------------------------------------------------------- #
# 2. Reboot jump + late backfill + slow clock drift
# --------------------------------------------------------------------------- #
def _reboot() -> Scenario:
    cal = [
        CalSourceSpec("C1", 50.0, 1000.0, [-2400.0 + 100.0 * i for i in range(64)]),
        CalSourceSpec("C2", 1950.0, -250.0, [-2350.0 + 105.0 * i for i in range(62)]),
    ]
    stations = [
        StationSpec("S1", 200.0, 250.0, 2600.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec(
            "S2",
            1250.0,
            100.0,
            1800.0,
            [
                Boot("b1", -3600.0, BOOT_BASE_LOCAL_S, 60.0),
                # Reboot at t=900 s: the clock jumps ~+1500 s and drifts the other way.
                Boot("b2", 900.0, BOOT_BASE_LOCAL_S + 900.0 + 1500.0, -45.0,
                     reboot_reason="power cycle; supercapacitor-backed RTC lost 1500 s"),
            ],
            (40.0, 100.0),
        ),
        StationSpec(
            "S3",
            1200.0,
            900.0,
            2600.0,
            [_boots_standard()],
            (1200.0, 1800.0),
            late=True,
            note="offline for 20 minutes, backfilled hours later",
        ),
        StationSpec("S4", 380.0, 880.0, 2600.0, [_boots_standard()], (20.0, 60.0)),
    ]
    sources = [
        SourceSpec("A", "whistle", 650.0, 250.0, [240.0, 700.0, 1500.0, 2400.0], [0.80, 0.60, 0.45]),
        SourceSpec("B", "whistle", 1150.0, 800.0, [1200.0, 3000.0], [0.62, 0.88, 0.30]),
    ]
    return Scenario(
        name="reboot",
        title="Station restart with a clock jump and a late backfill",
        description=(
            "S2 restarted at t=900 s; its clock jumped by 1500 s and its drift changed "
            "sign. S3 was offline and needs to upload records 20 minutes after the "
            "others. Arrival order and wall-clock elapsed time must not enter the "
            "association: only per-boot calibrated reference time counts, and a station "
            "without a complete progress report keeps its events provisional."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -400.0, "x_max": 2200.0, "y_min": -600.0, "y_max": 1400.0},
        expected={
            "events": 6,
            "rebooted_stations": ["S2"],
            "late_station": "S3",
            "note": "S2's two boots must have separate calibration models",
        },
    )


# --------------------------------------------------------------------------- #
# 3. Collinear stations: genuine mirror ambiguity
# --------------------------------------------------------------------------- #
def _collinear() -> Scenario:
    cal = [
        CalSourceSpec("C1", -50.0, 1200.0, [-2100.0 + 120.0 * i for i in range(32)]),
        CalSourceSpec("C2", 2250.0, -200.0, [-2050.0 + 115.0 * i for i in range(32)]),
    ]
    stations = [
        StationSpec("S1", 150.0, 500.0, 1900.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 750.0, 500.0, 1900.0, [_boots_standard()], (40.0, 100.0)),
        StationSpec("S3", 1350.0, 500.0, 1900.0, [_boots_standard()], (50.0, 110.0)),
        StationSpec("S4", 1950.0, 500.0, 1900.0, [_boots_standard()], (30.0, 90.0)),
    ]
    sources = [
        SourceSpec("A", "croak", 900.0, 180.0, [150.0, 800.0, 1450.0], [0.40, 1.00, 0.50]),
        SourceSpec("B", "croak", 900.0, 820.0, [420.0, 1100.0, 1750.0], [0.72, 0.55, 0.80]),
    ]
    return Scenario(
        name="collinear",
        title="Collinear stations cannot resolve the mirror pair",
        description=(
            "All four stations lie exactly on y = 500 m. Reflection across that line "
            "leaves every propagation delay unchanged, so a call at (900, 180) is "
            "timing-indistinguishable from one at (900, 820). The service must return "
            "both disconnected feasible regions and must not report one coordinate. "
            "Counting is unaffected: each call is still one event."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": 0.0, "x_max": 2200.0, "y_min": -400.0, "y_max": 1400.0},
        expected={
            "events": 6,
            "collinear": True,
            "min_regions_max": 6,
            "mirror_pair_y": [180.0, 820.0],
            "note": "every event must expose at least two feasible regions",
        },
    )


# --------------------------------------------------------------------------- #
# 4. Retransmission: identical replay + content conflict
# --------------------------------------------------------------------------- #
def _duplicates() -> Scenario:
    cal = [
        CalSourceSpec("C1", -100.0, 400.0, [-1500.0 + 120.0 * i for i in range(30)]),
        CalSourceSpec("C2", 1700.0, 800.0, [-1450.0 + 115.0 * i for i in range(30)]),
    ]
    stations = [
        StationSpec("S1", 150.0, 150.0, 1700.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1750.0, 200.0, 1700.0, [_boots_standard()], (40.0, 90.0)),
        StationSpec("S3", 1150.0, 1300.0, 1700.0, [_boots_standard()], (50.0, 120.0)),
        StationSpec("S4", 600.0, 900.0, 1500.0, [_boots_standard()], (30.0, 85.0)),
    ]
    sources = [
        SourceSpec("A", "drum", 650.0, 450.0, [200.0, 1200.0], [0.60, 0.30, 1.00]),
        SourceSpec("B", "drum", 900.0, 620.0, [1600.0], [0.62, 0.34, 1.00]),
    ]
    return Scenario(
        name="duplicates",
        title="Retransmitted batches: identical replay and a content conflict",
        description=(
            "After the normal uploads, station S1 re-sends its first batch byte-for-byte "
            "(must be counted as a duplicate and change nothing), then re-sends the "
            "record with source_seq 7 carrying a different uncertainty under the same "
            "sequence number (must land in the conflict table and leave the original "
            "untouched). Events and counts must be identical with and without the "
            "retransmission."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -200.0, "x_max": 2000.0, "y_min": -200.0, "y_max": 1600.0},
        feature_noise=0.006,
        expected={
            "events": 3,
            "replay_batch_index": 0,
            "conflict_station": "S1",
            "conflict_uncertainty_ms": 45.0,
        },
    )


# --------------------------------------------------------------------------- #
# 5. Late report + calibration revision => event revisions
# --------------------------------------------------------------------------- #
def _revision() -> Scenario:
    cal = [
        CalSourceSpec("C1", -100.0, 400.0, [-1800.0 + 110.0 * i for i in range(40)]),
        # Emissions of C2 after t=600 arrive only in the late phase, so S1's and
        # S2's calibration gains anchors between the two runs.
        CalSourceSpec("C2", 1700.0, 800.0, [-1780.0 + 105.0 * i for i in range(40)],
                      late_from_s=600.0),
    ]
    stations = [
        StationSpec("S1", 350.0, 150.0, 1600.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1300.0, 300.0, 1600.0, [_boots_standard()], (40.0, 90.0)),
        StationSpec(
            "S3",
            800.0,
            1000.0,
            1600.0,
            [_boots_standard()],
            (1200.0, 1600.0),
            late=True,
            note="slow link; whole boot uploaded in the late phase",
        ),
        StationSpec("S4", 550.0, 820.0, 1500.0, [_boots_standard()], (30.0, 95.0)),
    ]
    sources = [
        SourceSpec("A", "trill", 700.0, 430.0, [250.0, 800.0, 1400.0], [0.70, 0.40, 0.90]),
        SourceSpec("B", "trill", 950.0, 600.0, [1900.0, 2200.0], [0.90, 0.28, 0.65]),
    ]
    return Scenario(
        name="revision",
        title="Late report and calibration revision produce event revisions",
        description=(
            "The first run happens before S3's backfill and before the late calibration "
            "anchors arrive. It publishes provisional events without S3. The second run "
            "sees S3's records and a revised calibration, so events that now include S3 "
            "are published as revisions naming the events they supersede: no activity is "
            "silently split into two while the old count remains."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -300.0, "x_max": 2000.0, "y_min": -400.0, "y_max": 1400.0},
        feature_noise=0.008,
        expected={
            "events_first_run": 5,
            "superseded_by_second_run": 5,
            "late_station": "S3",
            "note": "second run's calibration hash for S1/S2 must differ from the first run",
        },
    )


# --------------------------------------------------------------------------- #
# 6. Fault attribution: clock offset, mis-recorded coordinate, speed change
# --------------------------------------------------------------------------- #
def _diagnosis() -> Scenario:
    cal = [
        # Sources spread around every station so directional sectors are filled.
        CalSourceSpec("C1", 50.0, -300.0, [-2400.0 + 120.0 * i for i in range(40)]),
        CalSourceSpec("C2", 2050.0, -300.0, [-2350.0 + 120.0 * i for i in range(40)]),
        CalSourceSpec("C3", 1050.0, 1300.0, [-2300.0 + 120.0 * i for i in range(40)]),
    ]
    # S1's clock steps +180 ms late in the boot.  The step stays under the
    # pipeline's single-segment RMS rejection threshold, so the calibration
    # model stays valid while the late anchors — which surround the station in
    # every direction — keep a same-sign residual the clock hypothesis must
    # explain and a coordinate shift cannot.
    stations = [
        StationSpec(
            "S1", 300.0, 100.0, 2600.0,
            [Boot("b1", -3600.0, BOOT_BASE_LOCAL_S, 35.0,
                  offset_step_s=1500.0, offset_step_us=180_000.0)],
            (25.0, 70.0),
            note="clock steps +180 ms late in boot b1; same-sign residual in every direction",
        ),
        StationSpec(
            "S2", 1400.0, 150.0, 2600.0, [_boots_standard()], (40.0, 100.0),
            registered_dx=-80.0,
            note="registered 80 m west of the true measurement point",
        ),
        StationSpec("S3", 1050.0, 950.0, 2600.0, [_boots_standard()], (55.0, 140.0)),
        StationSpec("S4", 550.0, 600.0, 2600.0, [_boots_standard()], (20.0, 60.0)),
    ]
    sources = [
        SourceSpec("A", "chirp", 760.0, 300.0, [120.0, 900.0, 1700.0], [1.0, 0.30, 0.20, 0.90]),
        SourceSpec("B", "chirp", 880.0, 480.0, [123.5, 905.0, 1695.0], [1.0, 0.42, 0.34, 0.86]),
    ]
    return Scenario(
        name="diagnosis",
        title="Clock offset, mis-recorded coordinate and regional speed change",
        description=(
            "Three distinct fault signatures coexist: S1's clock steps +180 ms late in "
            "the boot while the single-segment calibration stays valid, leaving a "
            "same-sign residual from every direction; S2 is registered 80 m from its "
            "true position (residual reverses with source direction); and the region "
            "really propagates at 328 m/s while the service baseline stays 343 m/s "
            "(residual grows with source distance). One grossly anomalous S3 anchor is "
            "injected so the diagnosis must document its exclusion before/after."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -400.0, "x_max": 2300.0, "y_min": -700.0, "y_max": 1500.0},
        propagation_speed_mps=328.0,
        anchor_outliers_ms={"S3": 900.0},
        anchor_noise_ms=2.0,
        expected={
            "clock_offset_station": "S1",
            "clock_offset_min_ms": 150.0,
            "coordinate_station": "S2",
            "coordinate_shift_min_m": 50.0,
            "regional_speed_between_mps": [320.0, 336.0],
            "outlier_station": "S3",
            "outlier_min_ms": 500.0,
            # The independent verifier's forward-model tolerance must cover the
            # intentionally injected 900 ms outlier anchor and 180 ms clock step.
            "forward_model_tolerance_ms": 950.0,
        },
    )


# --------------------------------------------------------------------------- #
# 7. Sparse anchors: no station may receive a certified parameter
# --------------------------------------------------------------------------- #
def _sparse_anchors() -> Scenario:
    # Two emissions per source: each station hears only four anchors in total,
    # fewer than the six needed to reserve a hold-out partition, so no
    # hypothesis can be certified even though a fit line exists.
    cal = [
        CalSourceSpec("C1", 100.0, -150.0, [-2400.0, -1200.0]),
        CalSourceSpec("C2", 1650.0, 900.0, [-2280.0, -1080.0]),
    ]
    stations = [
        StationSpec("S1", 300.0, 100.0, 2200.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1400.0, 150.0, 2200.0, [_boots_standard()], (40.0, 100.0)),
        StationSpec("S3", 1050.0, 950.0, 2200.0, [_boots_standard()], (55.0, 140.0)),
        StationSpec("S4", 550.0, 600.0, 2200.0, [_boots_standard()], (20.0, 60.0)),
    ]
    sources = [
        SourceSpec("A", "chirp", 760.0, 300.0, [120.0, 900.0], [1.0, 0.30, 0.20, 0.90]),
    ]
    return Scenario(
        name="sparse_anchors",
        title="Anchor density too low for any certifiable hypothesis",
        description=(
            "Every station hears only a handful of calibration sources, so no boot has "
            "the minimum fit anchors and no hold-out partition can be formed. The "
            "diagnosis must report insufficient anchors, every hypothesis must stay "
            "non-identifiable, and the certification book must reject publication rather "
            "than force the residual onto a single parameter."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -300.0, "x_max": 2100.0, "y_min": -500.0, "y_max": 1300.0},
        anchor_noise_ms=2.0,
        expected={
            "max_anchors_per_boot": 4,
            "expect_no_certifiable_hypothesis": True,
        },
    )

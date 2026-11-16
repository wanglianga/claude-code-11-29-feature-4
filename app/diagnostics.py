"""Calibration fault attribution: hypothesis diagnostics with hold-out validation.

The service's regular calibration (``app.calibration.py``) maps each station
clock to reference time but deliberately attributes *every* residual to that
clock.  Maintenance staff need to know whether the remaining residuals really
come from the clock (offset/drift), from a mis-recorded station coordinate, or
from a change of the regional propagation speed — otherwise the next
localisation run inherits a systematic bias.

This module therefore keeps three competing hypotheses in a register and
judges them on anchors that never participated in the fit:

* ``clock``       — residual = offset + drift * (t - t0), per station/boot;
* ``coordinate``  — residual = unit(source->station) . shift / speed;
* ``speed``       — residual = distance * (1/v' - 1/v), regional and shared.

Two parameter organisations are produced for the same anchor split and the
same batch of localisation events:

* ``per_station`` estimates clock and coordinate corrections independently per
  station (local residuals are easy to attribute and a station with sparse
  anchors can still be processed), at the risk of absorbing a regional
  propagation change into device parameters;
* ``regional`` first estimates one shared propagation speed and only then adds
  a local coordinate correction for stations whose directional anchor coverage
  is sufficient — cross-station consistent, but stations lacking coverage
  temporarily get no certified local parameter.

All timestamps are integer microseconds; residuals are microseconds.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import calibration as calib
from .config import Settings
from .models import (
    CalibrationModel,
    CalibrationSourceRecord,
    DiagnosticEvidence,
    DiagnosticRun,
    Event,
    EventObservation,
    HypothesisRecord,
    Observation,
    PropagationRegion,
    Station,
)
from .physics import distance
from .timeutil import iso, now_us

H_CLOCK = "clock"
H_COORDINATE = "coordinate"
H_COMBINED = "clock_coordinate"
H_SPEED = "propagation_speed"
H_BASELINE = "baseline"

ORG_PER_STATION = "per_station"
ORG_REGIONAL = "regional"


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiagAnchor:
    """One calibration anchor with every geometric quantity precomputed."""

    evidence_id: int
    observation_id: int
    station_id: str
    boot_id: str
    source_id: str
    source_x: float
    source_y: float
    station_x: float
    station_y: float
    emission_us: int
    distance_m: float
    direction_rad: float
    expected_arrival_us: float
    measured_arrival_us: float
    residual_us: float
    local_capture_us: int
    mapping_source: str = "pipeline_calibration_model"

    @property
    def ref_time_us(self) -> float:
        return self.expected_arrival_us

    @property
    def unit_x(self) -> float:
        return (self.station_x - self.source_x) / self.distance_m if self.distance_m > 0 else 0.0

    @property
    def unit_y(self) -> float:
        return (self.station_y - self.source_y) / self.distance_m if self.distance_m > 0 else 0.0


def _region_for(db: Session, region_id: str | None) -> PropagationRegion | None:
    if region_id is None:
        return None
    return db.get(PropagationRegion, region_id)


def _fit_from_model(model: CalibrationModel, cfg: Settings) -> calib.CalibrationFit:
    """Rebuild a clock fit from a persisted calibration model (pipeline mirror)."""
    segments = [
        calib.Segment(
            local_from_us=seg["local_from_us"],
            local_to_us=seg["local_to_us"],
            slope=seg["slope"],
            intercept_us=seg["intercept_us"],
            anchor_ids=seg.get("anchor_observation_ids", []),
            rms_residual_us=seg["rms_residual_us"],
            max_residual_us=seg["max_residual_us"],
            se_slope=seg.get("se_slope", 0.0),
            se_intercept_us=seg.get("se_intercept_us", 0.0),
        )
        for seg in (model.segments or [])
    ]
    fit = calib.CalibrationFit(
        station_id=model.station_id,
        boot_id=model.boot_id,
        valid=model.valid,
        reason=model.invalid_reason,
        segments=segments,
        anchor_ids=model.anchor_ids or [],
        anchor_count=model.anchor_count,
        rms_residual_us=model.rms_residual_us,
        max_residual_us=model.max_residual_us,
        base_uncertainty_us=model.uncertainty_us,
        local_min_us=model.local_anchor_min_us,
        local_max_us=model.local_anchor_max_us,
    )
    fit.params = model.params_json or {}
    return fit


def _baseline_speed(region: PropagationRegion | None, cfg: Settings) -> float:
    return region.baseline_speed_mps if region is not None else cfg.propagation_speed_mps


def build_evidence(
    db: Session,
    run: DiagnosticRun,
    *,
    region: PropagationRegion | None,
    cfg: Settings,
) -> tuple[list[DiagAnchor], int]:
    """Freeze the calibration evidence library for this diagnostic run.

    A station's local reading is mapped to reference time in one of two ways:
    the pipeline's validated per-boot clock model when it exists, or — when the
    pipeline rejected the calibration (often precisely because a coordinate
    error inflated its residuals) — a plain robust clock trend fitted inside
    the diagnostic layer and explicitly labelled as such.  Source positions,
    emission instants, the recorded measurement point and boot identity are
    snapshotted as independent rows so later register edits cannot alter the
    diagnosis.
    """
    speed = _baseline_speed(region, cfg)
    rows = list(
        db.execute(
            select(Observation).where(
                Observation.kind == "calibration",
                Observation.ingest_at_us <= run.cutoff_us,
            )
        ).scalars()
    )
    sources = {s.id: s for s in db.execute(select(CalibrationSourceRecord)).scalars()}
    stations = {s.id: s for s in db.execute(select(Station)).scalars()}
    if run.scope_type == "station" and run.station_id:
        rows = [r for r in rows if r.station_id == run.station_id]
    valid_models: dict[tuple[str, str], CalibrationModel] = {
        (m.station_id, m.boot_id): m
        for m in db.execute(
            select(CalibrationModel).where(CalibrationModel.valid.is_(True))
        ).scalars()
    }
    invalid_models: dict[tuple[str, str], CalibrationModel] = {
        (m.station_id, m.boot_id): m
        for m in db.execute(
            select(CalibrationModel).where(CalibrationModel.valid.is_(False))
        ).scalars()
    }
    groups: dict[tuple[str, str], list[Observation]] = {}
    for obs in rows:
        if obs.reference_emission_us is None:
            continue
        st = stations.get(obs.station_id)
        src = sources.get(obs.calibration_source_id or "")
        if st is None or src is None:
            continue
        if region is not None and not (
            region.x_min <= st.x <= region.x_max
            and region.y_min <= st.y <= region.y_max
        ):
            continue
        groups.setdefault((obs.station_id, obs.boot_id), []).append(obs)

    skipped_unmapped = 0
    mapping_report: dict[str, dict] = {}
    anchors: list[DiagAnchor] = []
    for (station_id, boot_id), group_rows in sorted(groups.items()):
        st = stations[station_id]
        model = valid_models.get((station_id, boot_id))
        invalid = invalid_models.get((station_id, boot_id))
        internal_trend: tuple[float, float] | None = None
        if model is not None:
            mapping_source = "pipeline_calibration_model"
            fit = _fit_from_model(model, cfg)
        else:
            mapping_source = "diagnostic_clock_trend"
            internal_trend = _fit_internal_clock_trend(group_rows, sources, st, speed)
            if internal_trend is None:
                skipped_unmapped += len(group_rows)
                continue
            fit = None
        mapping_report[f"{station_id}/{boot_id}"] = {
            "mapping_source": mapping_source,
            "pipeline_model_valid": model is not None,
            "pipeline_invalid_reason": invalid.invalid_reason if invalid else None,
            "anchor_count": len(group_rows),
            "internal_trend_available": internal_trend is not None,
        }
        for obs in group_rows:
            src = sources[obs.calibration_source_id]
            if mapping_source == "pipeline_calibration_model":
                mapped = calib.apply_calibration(fit, obs.local_capture_us, cfg)
                if mapped.reference_arrival_us is None:
                    skipped_unmapped += 1
                    continue
                measured = float(mapped.reference_arrival_us)
            else:
                slope, intercept = internal_trend
                measured = (obs.local_capture_us - intercept) / slope
            d = distance(st.x, st.y, src.x, src.y)
            delay_us = d / speed * 1_000_000.0
            expected = obs.reference_emission_us + delay_us
            direction = math.atan2(st.y - src.y, st.x - src.x)
            evidence = DiagnosticEvidence(
                diagnostic_run_id=run.id,
                station_id=obs.station_id,
                boot_id=obs.boot_id,
                observation_id=obs.id,
                calibration_source_id=src.id,
                source_x=src.x,
                source_y=src.y,
                station_x=st.x,
                station_y=st.y,
                emission_us=obs.reference_emission_us,
                distance_m=d,
                direction_rad=direction,
                propagation_us=delay_us,
                reference_arrival_us=expected,
                local_capture_us=obs.local_capture_us,
                detrended_residual_us=measured - expected,
                mapping_source=mapping_source,
                partition="fit",
                excluded=False,
                created_at_us=now_us(),
            )
            db.add(evidence)
            db.flush()
            anchors.append(
                DiagAnchor(
                    evidence_id=evidence.id,
                    observation_id=obs.id,
                    station_id=obs.station_id,
                    boot_id=obs.boot_id,
                    source_id=f"{src.id}@{iso(obs.reference_emission_us)}",
                    source_x=src.x,
                    source_y=src.y,
                    station_x=st.x,
                    station_y=st.y,
                    emission_us=obs.reference_emission_us,
                    distance_m=d,
                    direction_rad=direction,
                    expected_arrival_us=expected,
                    measured_arrival_us=measured,
                    residual_us=measured - expected,
                    local_capture_us=obs.local_capture_us,
                    mapping_source=mapping_source,
                )
            )
    run.diagnosis_json = run.diagnosis_json or {}
    run.diagnosis_json["clock_mapping"] = mapping_report
    return anchors, skipped_unmapped


def _fit_internal_clock_trend(
    rows: list[Observation],
    sources: dict[str, CalibrationSourceRecord],
    station: Station,
    speed: float,
) -> tuple[float, float] | None:
    """Plain OLS trend ``local = slope * expected_ref + intercept``.

    Used only to give the diagnostic layer reference arrivals when the
    pipeline's own calibration was rejected.  A single median-absolute-
    deviation guard keeps one grossly anomalous anchor from dominating the
    trend without deleting any evidence row.
    """
    points: list[tuple[float, int]] = []
    for obs in rows:
        src = sources.get(obs.calibration_source_id or "")
        if src is None or obs.reference_emission_us is None:
            continue
        d = distance(station.x, station.y, src.x, src.y)
        expected_ref = obs.reference_emission_us + d / speed * 1e6
        points.append((expected_ref, obs.local_capture_us))
    if len(points) < 3:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points], dtype=float)
    slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
    resid = ys - (slope * xs + intercept)
    mad = float(np.median(np.abs(resid - np.median(resid)))) or 1.0
    keep = np.abs(resid - np.median(resid)) <= 8.0 * mad
    if int(keep.sum()) >= 3 and int(keep.sum()) < len(points):
        slope, intercept = (float(v) for v in np.polyfit(xs[keep], ys[keep], 1))
    return slope, intercept


# --------------------------------------------------------------------------- #
# Deterministic hold-out split
# --------------------------------------------------------------------------- #
def partition_anchors(
    anchors: list[DiagAnchor], cfg: Settings
) -> tuple[list[DiagAnchor], list[DiagAnchor]]:
    """Stratified, deterministic split per (station, boot).

    Every boot contributes anchors to both partitions so the two parameter
    organisations are always scored on the same hold-out set.  The split is a
    fixed stride over reference-time order — no randomness, so reruns against
    the same cutoff reproduce the exact same fit/hold-out assignment.
    """
    fit: list[DiagAnchor] = []
    holdout: list[DiagAnchor] = []
    groups: dict[tuple[str, str], list[DiagAnchor]] = {}
    for a in anchors:
        groups.setdefault((a.station_id, a.boot_id), []).append(a)
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda a: a.ref_time_us)
        n = len(ordered)
        want_holdout = min(
            max(cfg.diag_min_holdout, int(round(n * cfg.diag_holdout_fraction))),
            max(0, n - cfg.diag_min_fit_anchors),
        )
        if n >= cfg.diag_min_fit_anchors + cfg.diag_min_holdout:
            stride = max(1, n // want_holdout) if want_holdout else 0
            chosen = set()
            i = stride // 2
            while len(chosen) < want_holdout and i < n:
                chosen.add(i)
                i += stride
            for idx, a in enumerate(ordered):
                (holdout if idx in chosen else fit).append(a)
        else:
            # Too few anchors to reserve the required hold-out: everything
            # stays in the fit partition and the run is flagged non-certifiable.
            fit.extend(ordered)
    return fit, holdout


# --------------------------------------------------------------------------- #
# Linear algebra helpers
# --------------------------------------------------------------------------- #
def _lsq(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least squares with design condition number and residual RMS.

    The condition number is computed on the *column-normalised* design so the
    physical units of each parameter (microseconds, seconds, metres) do not
    make a well-conditioned fit look singular.
    """
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coeffs
    rms = float(np.sqrt(np.mean((target - predicted) ** 2))) if len(target) else 0.0
    if design.shape[1] == 1:
        condition = 1.0
    else:
        norms = np.linalg.norm(design, axis=0)
        norms[norms == 0] = 1.0
        singular = np.linalg.svd(design / norms, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    return coeffs, rms, condition


def _json_safe(value):
    """Recursively replace non-finite floats so the value is valid JSONB.

    PostgreSQL's JSONB type rejects ``NaN``/``Infinity`` tokens; a singular
    least-squares design therefore surfaces as ``null`` instead of crashing the
    diagnostic run.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _error_metrics(values: list[float]) -> dict:
    if not values:
        return {"rmse_ms": None, "mae_ms": None, "max_ms": None, "bias_ms": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "rmse_ms": round(float(np.sqrt(np.mean(arr**2))) / 1000.0, 6),
        "mae_ms": round(float(np.mean(np.abs(arr))) / 1000.0, 6),
        "max_ms": round(float(np.max(np.abs(arr))) / 1000.0, 6),
        "bias_ms": round(float(np.mean(arr)) / 1000.0, 6),
        "n": len(values),
    }


# --------------------------------------------------------------------------- #
# Hypothesis fitters (pure functions)
# --------------------------------------------------------------------------- #
@dataclass
class FittedHypothesis:
    hypothesis_type: str
    organization: str
    station_id: str | None
    boot_id: str | None
    region_id: str | None
    params: dict
    fit_metrics: dict
    identifiability: dict
    identifiable: bool
    predict: object = field(repr=False, default=None)
    direction_analysis: dict | None = None


def _time_span_s(anchors: list[DiagAnchor]) -> float:
    if not anchors:
        return 0.0
    return (max(a.ref_time_us for a in anchors) - min(a.ref_time_us for a in anchors)) / 1e6


def _direction_sectors(anchors: list[DiagAnchor], sectors: int) -> dict:
    bins: dict[int, list[float]] = {}
    for a in anchors:
        idx = int((a.direction_rad + math.pi) / (2.0 * math.pi) * sectors) % sectors
        bins.setdefault(idx, []).append(a.residual_us)
    return {
        str(idx): {
            "anchor_count": len(vals),
            "mean_residual_ms": round(float(np.mean(vals)) / 1000.0, 6),
            "same_sign_fraction": round(
                max(sum(1 for v in vals if v >= 0), sum(1 for v in vals if v < 0)) / len(vals), 4
            ),
        }
        for idx, vals in sorted(bins.items())
    }


def direction_analysis(anchors: list[DiagAnchor], cfg: Settings) -> dict:
    """Characterise residual geometry relative to source direction.

    Same-sign residuals from every direction look like a clock offset; a sign
    reversal across directions looks like a position error; a residual that
    grows with source distance looks like a propagation-speed change.
    """
    sector_info = _direction_sectors(anchors, cfg.diag_direction_sectors)
    populated = [s for s in sector_info.values() if s["anchor_count"] > 0]
    positives = sum(1 for s in populated if s["mean_residual_ms"] > 0)
    negatives = sum(1 for s in populated if s["mean_residual_ms"] < 0)
    same_sign = max(positives, negatives) / len(populated) if populated else 0.0
    # A reversal means sectors of BOTH signs are well represented, i.e. the
    # residuals are not predominantly one sign — the natural complement of the
    # clock-offset condition.  A threshold above 0.6 for both signs can never
    # hold, so the test is whether the dominant sign falls short of the
    # clock-offset threshold.
    reversal = (
        len(populated) >= 2
        and positives >= 1
        and negatives >= 1
        and same_sign < cfg.diag_same_sign_fraction
    )
    distances = np.array([a.distance_m for a in anchors])
    residuals = np.array([a.residual_us for a in anchors])
    if len(anchors) >= 3 and float(np.std(distances)) > 0 and float(np.std(residuals)) > 0:
        corr = float(np.corrcoef(distances, residuals)[0, 1])
    else:
        corr = 0.0
    return {
        "sectors": sector_info,
        "populated_sector_count": len(populated),
        "same_sign_fraction": round(same_sign, 4),
        "direction_reversal": reversal,
        "distance_correlation": round(corr, 4),
        "clock_offset_plausible": same_sign >= cfg.diag_same_sign_fraction and not reversal,
        "single_clock_fix_inadequate": reversal,
    }


def fit_clock(
    anchors: list[DiagAnchor], organization: str, cfg: Settings,
    station_id: str | None = None, boot_id: str | None = None,
    region_id: str | None = None,
) -> FittedHypothesis:
    """Clock hypothesis: affine offset/drift, plus an optional step.

    A constant offset alone is mathematically indistinguishable from the
    intercept of the per-boot clock calibration, so the identifiable clock
    signature is a *step*: the residual jumps at one reference instant and
    stays at the new level on both sides.  Breakpoints are scanned over anchor
    quantiles with a shared drift and two offsets; the step is accepted only
    when it clears the noise floor and improves the fit materially.  Otherwise
    the model falls back to a plain affine offset/drift.
    """
    ordered = sorted(anchors, key=lambda a: a.ref_time_us)
    t0 = ordered[0].ref_time_us
    xs = np.array([(a.ref_time_us - t0) / 1e6 for a in ordered])
    ys = np.array([a.residual_us for a in ordered])
    span = _time_span_s(ordered)
    base_design = np.column_stack([np.ones(len(ordered)), xs])
    (offset, drift), base_rms, condition = _lsq(base_design, ys)

    best: dict | None = None
    min_side = max(3, cfg.diag_min_fit_anchors // 2)
    for cut in range(min_side, len(ordered) - min_side):
        indicator = (np.arange(len(ordered)) >= cut).astype(float)
        design = np.column_stack([np.ones(len(ordered)), xs, indicator])
        coeffs, rms, _cond = _lsq(design, ys)
        step = float(coeffs[2])
        gain = (base_rms - rms) / base_rms if base_rms > 0 else 0.0
        if abs(step) < cfg.diag_min_step_ms * 1000.0 or gain < cfg.diag_step_min_gain:
            continue
        if best is None or rms < best["rms"]:
            best = {
                "rms": rms, "step": step, "offset": float(coeffs[0]),
                "drift": float(coeffs[1]),
                "breakpoint_us": ordered[cut].ref_time_us,
                "gain": gain,
            }

    identifiable = (
        len(anchors) >= cfg.diag_min_fit_anchors
        and span >= cfg.diag_min_anchor_span_s
        and condition <= cfg.diag_max_design_condition
    )
    reasons = []
    if len(anchors) < cfg.diag_min_fit_anchors:
        reasons.append(f"need >= {cfg.diag_min_fit_anchors} fit anchors, have {len(anchors)}")
    if span < cfg.diag_min_anchor_span_s:
        reasons.append(f"anchor span {span:.0f}s < {cfg.diag_min_anchor_span_s:.0f}s")
    if condition > cfg.diag_max_design_condition:
        reasons.append(f"design condition {condition:.1f} > {cfg.diag_max_design_condition}")

    if best is not None:
        offset, drift, step, breakpoint_us = (
            best["offset"], best["drift"], best["step"], best["breakpoint_us"]
        )
        params = {
            "model": "affine_with_step",
            "offset_us": round(offset, 3),
            "offset_ms": round(offset / 1000.0, 6),
            "drift_ppm": round(drift, 6),
            "step_us": round(step, 3),
            "step_ms": round(step / 1000.0, 6),
            "breakpoint_ref_us": int(round(breakpoint_us)),
            "breakpoint_ref": iso(int(round(breakpoint_us))),
            "t0_us": int(round(t0)),
            "fit_gain_vs_affine": round(best["gain"], 4),
        }
        reasons = [r for r in reasons if "design condition" not in r] or ["ok"]
        identifiable = identifiable and abs(step) >= cfg.diag_min_step_ms * 1000.0

        def predict(a: DiagAnchor) -> float:
            jump = step if a.ref_time_us >= breakpoint_us else 0.0
            return offset + drift * (a.ref_time_us - t0) / 1e6 + jump

        fit_metrics = {
            "fit_rmse_ms": round(best["rms"] / 1000.0, 6),
            "affine_fit_rmse_ms": round(base_rms / 1000.0, 6),
            "design_condition": round(condition, 3),
            "fit_anchor_count": len(anchors),
            "time_span_s": round(span, 2),
            "step_detected": True,
        }
    else:
        params = {
            "model": "affine",
            "offset_us": round(float(offset), 3),
            "offset_ms": round(float(offset) / 1000.0, 6),
            "drift_us_per_s": round(float(drift), 6),
            "drift_ppm": round(float(drift), 6),
            "t0_us": int(round(t0)),
        }

        def predict(a: DiagAnchor) -> float:
            return float(offset) + float(drift) * (a.ref_time_us - t0) / 1e6

        fit_metrics = {
            "fit_rmse_ms": round(base_rms / 1000.0, 6),
            "design_condition": round(condition, 3),
            "fit_anchor_count": len(anchors),
            "time_span_s": round(span, 2),
            "step_detected": False,
        }

    return FittedHypothesis(
        H_CLOCK, organization, station_id, boot_id, region_id,
        params, fit_metrics,
        {"identifiable": identifiable, "conditions": reasons or ["ok"],
         "time_span_s": round(span, 2), "design_condition": round(condition, 3)},
        identifiable, predict,
    )


def fit_coordinate(
    anchors: list[DiagAnchor], speed: float, organization: str, cfg: Settings,
    station_id: str | None = None, boot_id: str | None = None,
    region_id: str | None = None,
) -> FittedHypothesis:
    """r_us * v / 1e6 = u_x*dx + u_y*dy for a shifted station measurement point."""
    design = np.array([[a.unit_x, a.unit_y] for a in anchors])
    # residual [us] * speed [m/s] / 1e6 -> metres of path-length error.
    target = np.array([a.residual_us * speed / 1e6 for a in anchors])
    (dx, dy), rms_m, condition = _lsq(design, target)
    rms_us = rms_m * 1e6 / speed
    shift = math.hypot(float(dx), float(dy))
    analysis = direction_analysis(anchors, cfg)
    distinct = analysis["populated_sector_count"]
    distances = [a.distance_m for a in anchors]
    spread = max(distances) - min(distances) if distances else 0.0
    identifiable = (
        len(anchors) >= cfg.diag_min_fit_anchors
        and distinct >= cfg.diag_min_directions
        and condition <= cfg.diag_max_design_condition
        and shift <= cfg.diag_max_coord_shift_m
        and spread >= cfg.diag_min_distance_spread_m
    )
    reasons = []
    if distinct < cfg.diag_min_directions:
        reasons.append(
            f"only {distinct} populated direction sectors, need {cfg.diag_min_directions}"
        )
    if condition > cfg.diag_max_design_condition:
        reasons.append(f"design condition {condition:.1f} > {cfg.diag_max_design_condition}")
    if shift > cfg.diag_max_coord_shift_m:
        reasons.append(f"implausible shift {shift:.1f} m > {cfg.diag_max_coord_shift_m:.0f} m")
    params = {
        "dx_m": round(float(dx), 4),
        "dy_m": round(float(dy), 4),
        "shift_m": round(shift, 4),
        "corrected_x": round(anchors[0].station_x + float(dx), 4) if anchors else None,
        "corrected_y": round(anchors[0].station_y + float(dy), 4) if anchors else None,
        "speed_mps": speed,
    }

    def predict(a: DiagAnchor) -> float:
        return (a.unit_x * float(dx) + a.unit_y * float(dy)) * 1e6 / speed

    return FittedHypothesis(
        H_COORDINATE, organization, station_id, boot_id, region_id,
        params,
        {"fit_rmse_ms": round(rms_us / 1000.0, 6), "fit_rmse_m": round(rms_m, 4),
         "design_condition": round(condition, 3), "fit_anchor_count": len(anchors)},
        {"identifiable": identifiable, "conditions": reasons or ["ok"],
         "direction_sectors": distinct, "design_condition": round(condition, 3),
         "distance_spread_m": round(spread, 2)},
        identifiable, predict, analysis,
    )


def fit_combined(
    anchors: list[DiagAnchor], speed: float, organization: str, cfg: Settings,
    station_id: str | None = None, boot_id: str | None = None,
    region_id: str | None = None,
) -> FittedHypothesis:
    """Clock offset/drift and station coordinate shift estimated together."""
    t0 = min(a.ref_time_us for a in anchors)
    scale = speed / 1e6
    design = np.array([
        [1.0, (a.ref_time_us - t0) / 1e6, a.unit_x * scale, a.unit_y * scale]
        for a in anchors
    ])
    target = np.array([a.residual_us for a in anchors])
    coeffs, rms, condition = _lsq(design, target)
    offset, drift, dx, dy = (float(v) for v in coeffs)
    shift = math.hypot(dx, dy)
    analysis = direction_analysis(anchors, cfg)
    identifiable = (
        len(anchors) >= cfg.diag_min_fit_anchors + 1
        and condition <= cfg.diag_max_design_condition
        and shift <= cfg.diag_max_coord_shift_m
        and analysis["populated_sector_count"] >= cfg.diag_min_directions
    )
    params = {
        "offset_us": round(offset, 3),
        "offset_ms": round(offset / 1000.0, 6),
        "drift_ppm": round(drift, 6),
        "t0_us": int(round(t0)),
        "dx_m": round(dx, 4),
        "dy_m": round(dy, 4),
        "shift_m": round(shift, 4),
        "speed_mps": speed,
    }

    def predict(a: DiagAnchor) -> float:
        return offset + drift * (a.ref_time_us - t0) / 1e6 + (
            a.unit_x * dx + a.unit_y * dy
        ) * scale

    return FittedHypothesis(
        H_COMBINED, organization, station_id, boot_id, region_id,
        params,
        {"fit_rmse_ms": round(rms / 1000.0, 6), "design_condition": round(condition, 3),
         "fit_anchor_count": len(anchors)},
        {"identifiable": identifiable,
         "conditions": ["ok"] if identifiable else [
             "combined clock+coordinate model needs more anchors or directional coverage"],
         "design_condition": round(condition, 3)},
        identifiable, predict, analysis,
    )


def fit_speed(
    anchors: list[DiagAnchor], baseline_speed: float, cfg: Settings,
    *, organization: str, station_intercepts: bool = False,
    station_id: str | None = None, region_id: str | None = None,
    boot_id: str | None = None,
) -> FittedHypothesis:
    """Shared regional speed: r_seconds = k * distance (+ per-station intercepts)."""
    station_keys = sorted({a.station_id for a in anchors})
    intercept_index = {sid: i for i, sid in enumerate(station_keys)}
    rows = []
    for a in anchors:
        row = [0.0] * (len(station_keys) + 1)
        if station_intercepts:
            row[intercept_index[a.station_id]] = 1.0
        else:
            row[0] = 1.0
        row[-1] = a.distance_m
        rows.append(row)
    design = np.array(rows)
    target = np.array([a.residual_us / 1e6 for a in anchors])  # seconds
    coeffs, rms_s, condition = _lsq(design, target)
    k = float(coeffs[-1])  # s/m
    inv_v = 1.0 / baseline_speed + k
    new_speed = 1.0 / inv_v if inv_v > 0 else math.inf
    distances = [a.distance_m for a in anchors]
    spread = max(distances) - min(distances) if distances else 0.0
    within_bounds = cfg.diag_speed_min_mps <= new_speed <= cfg.diag_speed_max_mps
    identifiable = (
        len(anchors) >= cfg.diag_min_fit_anchors
        and spread >= cfg.diag_min_distance_spread_m
        and within_bounds
        and math.isfinite(new_speed)
        and (len(station_keys) >= 2 if organization == ORG_REGIONAL else True)
    )
    reasons = []
    if spread < cfg.diag_min_distance_spread_m:
        reasons.append(f"distance spread {spread:.0f} m < {cfg.diag_min_distance_spread_m:.0f} m")
    if not within_bounds:
        reasons.append(f"fitted speed {new_speed:.2f} m/s outside plausible bounds")
    if organization == ORG_REGIONAL and len(station_keys) < 2:
        reasons.append("regional speed needs anchors from at least two stations")
    intercepts = {
        sid: round(float(coeffs[intercept_index[sid]]) * 1e6, 3)
        for sid in station_keys
    } if station_intercepts else {}
    params = {
        "speed_mps": round(new_speed, 5),
        "baseline_speed_mps": baseline_speed,
        "speed_delta_mps": round(new_speed - baseline_speed, 5),
        "speed_delta_fraction": round((new_speed - baseline_speed) / baseline_speed, 6),
        "k_seconds_per_m": k,
        "station_clock_offsets_us": intercepts,
    }

    def predict(a: DiagAnchor) -> float:
        base = k * a.distance_m * 1e6
        if station_intercepts:
            base += float(coeffs[intercept_index[a.station_id]]) * 1e6
        return base

    return FittedHypothesis(
        H_SPEED, organization, station_id, boot_id, region_id,
        params,
        {"fit_rmse_ms": round(rms_s * 1000.0, 6), "design_condition": round(condition, 3),
         "fit_anchor_count": len(anchors), "station_count": len(station_keys),
         "distance_spread_m": round(spread, 2)},
        {"identifiable": identifiable, "conditions": reasons or ["ok"],
         "distance_spread_m": round(spread, 2), "station_count": len(station_keys)},
        identifiable, predict,
    )


def evaluate_on_holdout(hyp: FittedHypothesis, holdout: list[DiagAnchor]) -> dict:
    if not holdout:
        return {**_error_metrics([]), "anchor_ids": [], "per_anchor": []}
    errors = [a.residual_us - hyp.predict(a) for a in holdout]
    metrics = _error_metrics(errors)
    metrics["anchor_ids"] = [a.evidence_id for a in holdout]
    metrics["per_anchor"] = [
        {
            "evidence_id": a.evidence_id,
            "station_id": a.station_id,
            "boot_id": a.boot_id,
            "residual_ms": round(a.residual_us / 1000.0, 6),
            "predicted_ms": round(hyp.predict(a) / 1000.0, 6),
            "error_ms": round((a.residual_us - hyp.predict(a)) / 1000.0, 6),
            "distance_m": round(a.distance_m, 2),
            "direction_rad": round(a.direction_rad, 4),
        }
        for a in holdout
    ]
    return metrics


def baseline_holdout_metrics(holdout: list[DiagAnchor]) -> dict:
    metrics = _error_metrics([a.residual_us for a in holdout])
    metrics["anchor_ids"] = [a.evidence_id for a in holdout]
    return metrics


# --------------------------------------------------------------------------- #
# Outlier rejection (fit partition only; hold-out is never touched)
# --------------------------------------------------------------------------- #
def reject_outliers(
    fit_anchors: list[DiagAnchor], holdout: list[DiagAnchor], cfg: Settings,
    speed: float,
) -> tuple[list[DiagAnchor], list[dict], dict]:
    """Iterative sigma clipping on the richest single-station model.

    Returns the cleaned fit set, per-anchor exclusion records and a
    before/after comparison.  Hold-out anchors are immutable here, so the
    published before/after numbers both come from the same untouched set.
    """
    if len(fit_anchors) < cfg.diag_min_fit_anchors + 1:
        return list(fit_anchors), [], {"performed": False,
                                       "reason": "not enough fit anchors for sigma clipping"}
    trial = fit_combined(fit_anchors, speed, ORG_PER_STATION, cfg)
    errors = np.array([a.residual_us - trial.predict(a) for a in fit_anchors])
    sigma = float(np.std(errors)) or 1.0
    threshold = cfg.diag_outlier_sigma * sigma
    dropped = [
        (a, float(err)) for a, err in zip(fit_anchors, errors) if abs(err) > threshold
    ]
    if not dropped:
        return list(fit_anchors), [], {
            "performed": True,
            "excluded_count": 0,
            "sigma_ms": round(sigma / 1000.0, 4),
            "threshold_ms": round(threshold / 1000.0, 4),
        }
    dropped_ids = {a.evidence_id for a, _ in dropped}
    cleaned = [a for a in fit_anchors if a.evidence_id not in dropped_ids]
    before = baseline_holdout_metrics(holdout)
    records = [
        {
            "evidence_id": a.evidence_id,
            "observation_id": a.observation_id,
            "station_id": a.station_id,
            "boot_id": a.boot_id,
            "source_id": a.source_id,
            "residual_ms": round(a.residual_us / 1000.0, 4),
            "model_error_ms": round(err / 1000.0, 4),
            "reason": (
                f"outlier: |model error| {abs(err) / 1000.0:.1f} ms > "
                f"{cfg.diag_outlier_sigma} sigma ({threshold / 1000.0:.1f} ms)"
            ),
        }
        for a, err in dropped
    ]
    comparison = {
        "performed": True,
        "excluded_count": len(dropped),
        "sigma_ms": round(sigma / 1000.0, 4),
        "threshold_ms": round(threshold / 1000.0, 4),
        "holdout_before_exclusion": before,
        "holdout_anchor_count": len(holdout),
        "fit_anchor_count_before": len(fit_anchors),
        "fit_anchor_count_after": len(cleaned),
    }
    return cleaned, records, comparison


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def _hypothesis_code(run: DiagnosticRun, hyp_type: str, suffix: str = "") -> str:
    base = f"H-{run.id}-{hyp_type}"
    return f"{base}-{suffix}" if suffix else base


def persist_hypothesis(
    db: Session, run: DiagnosticRun, hyp: FittedHypothesis,
    holdout_metrics: dict, rank: int | None, reason: str | None,
    code_suffix: str = "", version_hash: str | None = None,
) -> HypothesisRecord:
    row = HypothesisRecord(
        code=_hypothesis_code(run, hyp.hypothesis_type, code_suffix),
        diagnostic_run_id=run.id,
        organization=hyp.organization,
        hypothesis_type=hyp.hypothesis_type,
        station_id=hyp.station_id,
        boot_id=hyp.boot_id,
        region_id=hyp.region_id,
        status="candidate",
        preference_rank=rank,
        preference_reason=reason,
        params_json=_json_safe(hyp.params),
        fit_metrics_json=_json_safe(hyp.fit_metrics),
        holdout_metrics_json=_json_safe(holdout_metrics),
        identifiable=hyp.identifiable,
        identifiability_json=_json_safe(hyp.identifiability),
        direction_analysis_json=_json_safe(hyp.direction_analysis),
        calibration_version_hash=version_hash,
        created_at_us=now_us(),
    )
    db.add(row)
    db.flush()
    return row


# --------------------------------------------------------------------------- #
# Event validation: both organisations scored on the same localisation events
# --------------------------------------------------------------------------- #
def evaluate_events(
    db: Session,
    per_station_params: dict[tuple[str, str], dict],
    regional_params: dict,
    baseline_speed: float,
) -> list[dict]:
    """Recompute per-event arrival dispersion under each parameter organisation.

    The event's published representative point is held fixed, so the three
    numbers (baseline / per-station / regional) describe the same events under
    different parameter choices, not different localisations.
    """
    events = list(
        db.execute(
            select(Event).where(Event.status != "superseded").order_by(Event.id)
        ).scalars()
    )
    out = []
    for event in events:
        links = list(
            db.execute(
                select(EventObservation).where(EventObservation.event_id == event.id)
            ).scalars()
        )
        stations = {s.id: s for s in db.execute(select(Station)).scalars()}
        rows = []
        for link in links:
            st = stations.get(link.station_id)
            obs = db.get(Observation, link.observation_id)
            if st is None or obs is None:
                continue
            rows.append((st.x, st.y, float(link.corrected_arrival_us),
                         link.station_id, obs.boot_id))
        if len(rows) < 2:
            continue

        def dispersion(speed: float, coord_shift: dict, clock_corr: dict,
                       station_clock: dict | None = None) -> dict:
            station_clock = station_clock or {}
            adjusted = []
            for x, y, arrival, sid, boot in rows:
                shift = coord_shift.get(sid, {"dx_m": 0.0, "dy_m": 0.0})
                d = math.hypot(
                    x + shift["dx_m"] - event.x, y + shift["dy_m"] - event.y
                )
                corr = clock_corr.get((sid, boot), station_clock.get(sid, 0.0))
                adjusted.append((arrival - corr) - d / speed * 1e6)
            # Residuals about the implied common emission instant; the absolute
            # emission value itself is not an error metric.
            mean = sum(adjusted) / len(adjusted)
            residuals = [v - mean for v in adjusted]
            rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
            return {"rms_ms": round(rms / 1000.0, 4),
                    "spread_ms": round((max(adjusted) - min(adjusted)) / 1000.0, 4),
                    "implied_emission": iso(int(round(mean)))}

        ps_coord: dict[str, dict] = {}
        ps_clock: dict[tuple[str, str], float] = {}
        for (sid, boot), pack in per_station_params.items():
            coord = pack.get("coordinate") or {}
            if coord:
                ps_coord[sid] = coord
            ps_clock[(sid, boot)] = float(pack.get("clock", {}).get("offset_us", 0.0))
        regional_speed = float(regional_params.get("speed_mps", baseline_speed))
        if not math.isfinite(regional_speed):
            regional_speed = baseline_speed
        regional_coord = regional_params.get("coordinate_by_station", {})
        regional_station_clock = regional_params.get("station_clock_offsets_us", {})
        out.append({
            "event_id": event.id,
            "event_revision": event.revision,
            "event_status": event.status,
            "station_count": len(rows),
            "baseline": dispersion(baseline_speed, {}, {}),
            "per_station": dispersion(baseline_speed, ps_coord, ps_clock),
            "regional": dispersion(
                regional_speed, regional_coord, {}, regional_station_clock
            ),
        })
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _materially_better(candidate: float | None, baseline: float | None, cfg: Settings) -> bool:
    if candidate is None or baseline is None:
        return False
    if baseline <= 0:
        return candidate < baseline
    improvement = (baseline - candidate) / baseline
    return improvement >= cfg.diag_holdout_improvement_ratio


def _rank_hypotheses(
    candidates: list[tuple[FittedHypothesis, dict, str]],
    baseline_metrics: dict,
    cfg: Settings,
    direction: dict | None = None,
) -> list[tuple[int | None, str]]:
    """Rank competing hypotheses with physical-signature precedence.

    Pure hold-out RMSE cannot separate faults that project onto overlapping
    basis functions (a regional speed gradient and a clock step both raise the
    RMS), so the directional/temporal rules from the task statement take
    precedence among *eligible* (identifiable, within the regression envelope)
    hypotheses:

    1. a temporal clock *step* — no geometric or speed model can produce a jump;
    2. a residual that reverses with source direction — a single clock fix is
       explicitly inadequate, so a coordinate shift is preferred;
    3. a residual strongly correlated with source distance and a non-trivial
       fitted speed delta — a propagation-speed change;
    4. otherwise the lowest hold-out RMSE that improves materially wins.
    """
    direction = direction or {}
    baseline_rms = baseline_metrics.get("rmse_ms")
    entries: list[dict] = []
    for idx, (hyp, metrics, reason) in enumerate(candidates):
        rms = metrics.get("rmse_ms")
        eligible = (
            hyp.identifiable
            and rms is not None
            and baseline_rms is not None
            and rms <= baseline_rms * cfg.cert_holdout_worsen_ratio
            and rms <= cfg.diag_holdout_abs_limit_ms
        )
        entries.append({"idx": idx, "hyp": hyp, "rms": rms, "reason": reason,
                        "eligible": eligible})
    by_type = {e["hyp"].hypothesis_type: e for e in entries if e["eligible"]}
    chosen: dict | None = None
    clock_e = by_type.get(H_CLOCK)
    coord_e = by_type.get(H_COORDINATE)
    speed_e = by_type.get(H_SPEED)

    if clock_e is not None and clock_e["hyp"].params.get("model") == "affine_with_step":
        chosen = clock_e
        chosen["rule_reason"] = (
            "temporal clock step at "
            f"{clock_e['hyp'].params.get('breakpoint_ref')}: a jump in time has no "
            "geometric or propagation-speed explanation, so the clock hypothesis is "
            "evaluated first"
        )
    elif direction.get("single_clock_fix_inadequate") and coord_e is not None:
        chosen = coord_e
        chosen["rule_reason"] = (
            "residual sign reverses across source directions; a single clock correction "
            "cannot cover this pattern, so a coordinate shift is preferred"
        )
    elif (
        speed_e is not None
        and abs(direction.get("distance_correlation", 0.0)) >= cfg.diag_distance_correlation
        and abs(speed_e["hyp"].params.get("speed_delta_fraction", 0.0)) >= 0.01
    ):
        chosen = speed_e
        chosen["rule_reason"] = (
            f"residuals correlate with source distance (r="
            f"{direction.get('distance_correlation')}) and the fitted speed changes by "
            f"{speed_e['hyp'].params.get('speed_delta_fraction')}: propagation-speed change"
        )
    else:
        improved = [
            e for e in entries
            if e["eligible"] and _materially_better(e["rms"], baseline_rms, cfg)
        ]
        if improved:
            chosen = min(improved, key=lambda e: (e["rms"], len(e["hyp"].params)))
            chosen["rule_reason"] = "lowest hold-out RMSE with a material improvement"

    ranking: dict[int, tuple[int | None, str]] = {}
    if chosen is not None:
        ranking[chosen["idx"]] = (1, chosen.get("rule_reason") or chosen["reason"])
        rest = sorted(
            (e for e in entries if e["eligible"] and e["idx"] != chosen["idx"]),
            key=lambda e: e["rms"],
        )
        for rank, e in enumerate(rest, start=2):
            ranking[e["idx"]] = (rank, "eligible but dominated by the preferred signature")
    for e in entries:
        if e["idx"] in ranking:
            continue
        if not e["hyp"].identifiable:
            why = "parameters not identifiable: " + "; ".join(
                e["hyp"].identifiability.get("conditions", ["unknown"])
            )
        elif not e["eligible"]:
            why = (
                f"hold-out RMSE {e['rms']} ms is outside the regression envelope of "
                f"baseline {baseline_rms} ms"
            )
        else:
            why = "no distinctive physical signature and no material hold-out improvement"
        ranking[e["idx"]] = (None, why)
    return [ranking[i] for i in range(len(candidates))]


def create_diagnostic_run(
    db: Session,
    *,
    scope_type: str,
    organization: str,
    cfg: Settings,
    station_id: str | None = None,
    region_id: str | None = None,
    created_by: str | None = None,
) -> DiagnosticRun:
    if scope_type == "station":
        if not station_id or not db.get(Station, station_id):
            raise KeyError("station")
    if region_id is not None and not db.get(PropagationRegion, region_id):
        raise KeyError("region")
    run = DiagnosticRun(
        batch_id=str(uuid.uuid4()),
        scope_type=scope_type,
        station_id=station_id,
        region_id=region_id,
        organization=organization,
        cutoff_us=now_us(),
        baseline_speed_mps=_baseline_speed(_region_for(db, region_id), cfg),
        boot_ids=[],
        station_ids=[],
        status="running",
        created_by=created_by,
        created_at_us=now_us(),
    )
    db.add(run)
    db.flush()
    return run


def run_diagnosis(db: Session, run: DiagnosticRun, cfg: Settings) -> DiagnosticRun:
    region = _region_for(db, run.region_id)
    speed = _baseline_speed(region, cfg)
    anchors, skipped_unmapped = build_evidence(db, run, region=region, cfg=cfg)
    fit_anchors, holdout = partition_anchors(anchors, cfg)
    for a in fit_anchors:
        db.get(DiagnosticEvidence, a.evidence_id).partition = "fit"
    for a in holdout:
        db.get(DiagnosticEvidence, a.evidence_id).partition = "holdout"

    fit_anchors, exclusions, exclusion_comparison = reject_outliers(
        fit_anchors, holdout, cfg, speed
    )
    excluded_rows = []
    for rec in exclusions:
        row = db.get(DiagnosticEvidence, rec["evidence_id"])
        row.excluded = True
        row.exclusion_reason = rec["reason"]
        excluded_rows.append(rec)

    run.fit_anchor_count = len(fit_anchors)
    run.holdout_anchor_count = len(holdout)
    run.excluded_anchor_ids = [rec["evidence_id"] for rec in excluded_rows]
    run.boot_ids = sorted({a.boot_id for a in anchors})
    run.station_ids = sorted({a.station_id for a in anchors})

    enough_holdout = len(holdout) >= cfg.diag_min_holdout
    baseline_metrics = baseline_holdout_metrics(holdout)
    diagnosis: dict = {
        "scope": {"type": run.scope_type, "station_id": run.station_id,
                  "region_id": run.region_id},
        "baseline_speed_mps": speed,
        "clock_mapping": dict(run.diagnosis_json or {}).get("clock_mapping", {}),
        "partition": {
            "fit_anchor_count": len(fit_anchors),
            "holdout_anchor_count": len(holdout),
            "holdout_strategy": "deterministic stratified stride per station/boot",
            "enough_holdout": enough_holdout,
            "anchors_skipped_without_valid_clock_model": skipped_unmapped,
        },
        "baseline_holdout": baseline_metrics,
        "outlier_exclusion": exclusion_comparison,
        "excluded_anchors": excluded_rows,
        "organizations": {},
    }
    if not enough_holdout:
        diagnosis["status_reason"] = (
            f"insufficient_anchors: {len(holdout)} hold-out anchor(s), need at least "
            f"{cfg.diag_min_holdout}; hypotheses stay non-certifiable"
        )

    per_station_params: dict[tuple[str, str], dict] = {}
    station_summaries: list[dict] = []
    groups = sorted({(a.station_id, a.boot_id) for a in fit_anchors})
    for sid, boot in groups:
        s_fit = [a for a in fit_anchors if a.station_id == sid and a.boot_id == boot]
        s_hold = [a for a in holdout if a.station_id == sid and a.boot_id == boot]
        if len(s_fit) < cfg.diag_min_fit_anchors:
            station_summaries.append({
                "station_id": sid, "boot_id": boot,
                "status": "insufficient_anchors",
                "fit_anchor_count": len(s_fit),
                "holdout_anchor_count": len(s_hold),
                "reason": f"need >= {cfg.diag_min_fit_anchors} fit anchors, have {len(s_fit)}",
            })
            continue
        clock_hyp = fit_clock(s_fit, ORG_PER_STATION, cfg, sid, boot, run.region_id)
        coord_hyp = fit_coordinate(s_fit, speed, ORG_PER_STATION, cfg, sid, boot, run.region_id)
        combined_hyp = fit_combined(s_fit, speed, ORG_PER_STATION, cfg, sid, boot, run.region_id)
        speed_hyp = fit_speed(s_fit, speed, cfg, organization=ORG_PER_STATION,
                              station_id=sid, region_id=run.region_id, boot_id=boot)
        # One shared directional picture so every hypothesis is judged on the
        # same residual geometry rather than recomputing slightly different views.
        shared_direction = direction_analysis(s_fit, cfg)
        clock_hyp.direction_analysis = shared_direction
        speed_hyp.direction_analysis = shared_direction
        combined_hyp.direction_analysis = shared_direction
        hyps = [clock_hyp, coord_hyp, combined_hyp, speed_hyp]
        metrics = [evaluate_on_holdout(h, s_hold) for h in hyps]
        reasons = _preference_reasons(hyps, cfg)
        ranks = _rank_hypotheses(
            list(zip(hyps, metrics, reasons)),
            baseline_metrics_for(s_hold), cfg, shared_direction,
        )
        saved = []
        for hyp, hm, (rank, reason) in zip(hyps, metrics, ranks):
            row = persist_hypothesis(
                db, run, hyp, hm, rank, reason,
                code_suffix=f"{sid}-{boot}",
            )
            saved.append({"hypothesis_id": row.id, "type": hyp.hypothesis_type,
                          "rank": rank, "reason": reason, "holdout": hm,
                          "identifiable": hyp.identifiable,
                          "identifiability": hyp.identifiability})
        top = next((h for h, r in zip(hyps, ranks) if r[0] == 1), None)
        per_station_params[(sid, boot)] = {
            "clock": clock_hyp.params,
            "coordinate": coord_hyp.params if coord_hyp.identifiable else {},
            "combined": combined_hyp.params,
            "speed_single_station": speed_hyp.params,
        }
        station_summaries.append({
            "station_id": sid,
            "boot_id": boot,
            "status": "diagnosed",
            "fit_anchor_count": len(s_fit),
            "holdout_anchor_count": len(s_hold),
            "direction_analysis": shared_direction,
            "hypotheses": saved,
            "top_hypothesis": top.hypothesis_type if top else None,
        })

    diagnosis["organizations"][ORG_PER_STATION] = {
        "stations": station_summaries,
        "cross_station_note": (
            "per-station fits are not constrained to a shared propagation model; when anchors "
            "are sparse a regional speed change can be absorbed into clock/coordinate parameters"
        ),
    }

    # Regional organisation: shared speed first, then local corrections.
    regional: dict = {"station_corrections": [], "stations_without_correction": []}
    if len(fit_anchors) >= cfg.diag_min_fit_anchors:
        speed_hyp = fit_speed(
            fit_anchors, speed, cfg, organization=ORG_REGIONAL,
            station_intercepts=True, region_id=run.region_id,
        )
        speed_holdout = evaluate_on_holdout(speed_hyp, holdout)
        speed_rank, speed_reason = _rank_hypotheses(
            [(speed_hyp, speed_holdout, "shared regional speed with per-station clock offsets")],
            baseline_metrics, cfg,
        )[0]
        speed_row = persist_hypothesis(
            db, run, speed_hyp, speed_holdout, speed_rank, speed_reason,
            code_suffix="regional",
        )
        regional["speed_hypothesis_id"] = speed_row.id
        regional["speed"] = speed_hyp.params
        regional["speed_holdout"] = speed_holdout
        regional["speed_identifiable"] = speed_hyp.identifiable

        # Stage B: local coordinate corrections on speed-detrended residuals,
        # only where directional coverage supports identifiability.
        coord_by_station: dict[str, dict] = {}
        for sid in sorted({a.station_id for a in fit_anchors}):
            s_fit = [a for a in fit_anchors if a.station_id == sid]
            s_hold = [a for a in holdout if a.station_id == sid]
            detrended = [
                _detrended(a, speed_hyp) for a in s_fit
            ]
            if len(detrended) < cfg.diag_min_fit_anchors:
                regional["stations_without_correction"].append({
                    "station_id": sid,
                    "reason": f"insufficient anchors after regional fit ({len(detrended)})",
                })
                continue
            local_coord = fit_coordinate(
                detrended, speed_hyp.params["speed_mps"], ORG_REGIONAL, cfg,
                station_id=sid, region_id=run.region_id,
            )
            local_holdout = evaluate_on_holdout(
                local_coord, [_detrended(a, speed_hyp) for a in s_hold]
            )
            if local_coord.identifiable and enough_holdout:
                row = persist_hypothesis(
                    db, run, local_coord, local_holdout, None,
                    "local correction after shared regional speed",
                    code_suffix=f"regional-{sid}",
                )
                coord_by_station[sid] = local_coord.params
                regional["station_corrections"].append({
                    "hypothesis_id": row.id,
                    "station_id": sid,
                    "params": local_coord.params,
                    "holdout": local_holdout,
                })
            else:
                regional["stations_without_correction"].append({
                    "station_id": sid,
                    "reason": "; ".join(local_coord.identifiability["conditions"]),
                })
        regional["coordinate_by_station"] = coord_by_station
        regional["clock_offsets_us"] = {
            key: val for key, val in speed_hyp.params["station_clock_offsets_us"].items()
        }
    diagnosis["organizations"][ORG_REGIONAL] = regional

    # Cross-station consistency of standalone speed estimates.
    standalone_speeds = []
    for sid in sorted({a.station_id for a in fit_anchors}):
        s_fit = [a for a in fit_anchors if a.station_id == sid]
        if len(s_fit) >= cfg.diag_min_fit_anchors:
            h = fit_speed(
                s_fit, speed, cfg, organization=ORG_PER_STATION, station_id=sid,
                boot_id=s_fit[0].boot_id,
            )
            standalone_speeds.append({"station_id": sid,
                                      "speed_mps": h.params["speed_mps"],
                                      "identifiable": h.identifiable})
    if standalone_speeds:
        vals = [s["speed_mps"] for s in standalone_speeds if math.isfinite(s["speed_mps"])]
        diagnosis["cross_station_speed_consistency"] = {
            "per_station_speed_estimates": standalone_speeds,
            "spread_mps": round(max(vals) - min(vals), 4) if vals else None,
            "interpretation": (
                "wide spread with poor per-station direction coverage indicates regional "
                "propagation changes are being absorbed into device parameters"
            ),
        }

    diagnosis["event_validation"] = evaluate_events(
        db,
        per_station_params,
        {
            "speed_mps": regional.get("speed", {}).get("speed_mps", speed),
            "coordinate_by_station": regional.get("coordinate_by_station", {}),
            "station_clock_offsets_us": dict(
                regional.get("clock_offsets_us", {})
            ),
        },
        speed,
    )
    diagnosis["organization_recommendation"] = _recommend_organization(diagnosis, cfg)
    run.diagnosis_json = _json_safe(diagnosis)
    run.status = "completed"
    run.finished_at_us = now_us()
    db.flush()
    return run


def baseline_metrics_for(holdout: list[DiagAnchor]) -> dict:
    return baseline_holdout_metrics(holdout)


def _detrended(a: DiagAnchor, speed_hyp: FittedHypothesis) -> DiagAnchor:
    residual = a.residual_us - speed_hyp.predict(a)
    return DiagAnchor(
        evidence_id=a.evidence_id, observation_id=a.observation_id, station_id=a.station_id,
        boot_id=a.boot_id, source_id=a.source_id, source_x=a.source_x, source_y=a.source_y,
        station_x=a.station_x, station_y=a.station_y, emission_us=a.emission_us,
        distance_m=a.distance_m, direction_rad=a.direction_rad,
        expected_arrival_us=a.expected_arrival_us, measured_arrival_us=a.measured_arrival_us,
        residual_us=residual, local_capture_us=a.local_capture_us,
    )


def _preference_reasons(hyps: list[FittedHypothesis], cfg: Settings) -> list[str]:
    reasons = []
    for hyp in hyps:
        analysis = hyp.direction_analysis or {}
        if hyp.hypothesis_type == H_CLOCK:
            if analysis.get("single_clock_fix_inadequate"):
                reasons.append(
                    "direction_reversal: residuals change sign across source directions; "
                    "a single clock correction cannot cover this pattern"
                )
            elif analysis.get("clock_offset_plausible"):
                reasons.append(
                    "same-sign residuals from every direction: clock offset evaluated first"
                )
            else:
                reasons.append("clock offset/drift candidate")
        elif hyp.hypothesis_type == H_COORDINATE:
            if analysis.get("direction_reversal"):
                reasons.append(
                    "residual sign reverses with source direction: coordinate shift is the "
                    "physically consistent explanation"
                )
            else:
                reasons.append("coordinate shift candidate")
        elif hyp.hypothesis_type == H_SPEED:
            if abs(analysis.get("distance_correlation", 0.0)) >= cfg.diag_distance_correlation:
                reasons.append(
                    f"residuals correlate with source distance (r="
                    f"{analysis['distance_correlation']}): propagation speed candidate"
                )
            else:
                reasons.append("single-station speed fit (regional absorption risk if sparse)")
        else:
            reasons.append("combined clock + coordinate candidate")
    return reasons


def _recommend_organization(diagnosis: dict, cfg: Settings) -> dict:
    """Choose an organisation from anchor density, direction and consistency."""
    per_station = diagnosis["organizations"].get(ORG_PER_STATION, {}).get("stations", [])
    regional = diagnosis["organizations"].get(ORG_REGIONAL, {})
    sparse = [
        s for s in per_station if s.get("status") != "diagnosed"
        or any(not h.get("identifiable") for h in s.get("hypotheses", []))
    ]
    no_local = regional.get("stations_without_correction", [])
    consistency = diagnosis.get("cross_station_speed_consistency", {})
    speed_ok = regional.get("speed_identifiable", False)
    # No usable evidence at all: neither organisation can be recommended rather
    # than pretending per-station processing is a positive choice.
    if not per_station and not speed_ok:
        choice = "none"
        reason = (
            "no station produced usable anchors; neither parameter organisation is "
            "applicable until more calibration evidence is available"
        )
    elif speed_ok and not no_local:
        choice = ORG_REGIONAL
        reason = "shared speed is identifiable and every station has directional coverage"
    elif speed_ok and no_local:
        choice = ORG_REGIONAL
        reason = (
            "shared speed is identifiable; stations without local corrections must wait for "
            "more anchors rather than receive unverifiable device parameters"
        )
    elif sparse and not speed_ok:
        choice = ORG_PER_STATION
        reason = "anchors are too sparse for a shared regional speed; per-station processing only"
    else:
        spread = consistency.get("spread_mps")
        choice = ORG_REGIONAL if speed_ok else ORG_PER_STATION
        reason = (
            f"fallback decision; standalone speed spread {spread} m/s and regional "
            f"identifiability={speed_ok}"
        )
    return {
        "recommended": choice,
        "reason": reason,
        "anchor_density": {
            "stations_diagnosed": len(per_station) - len(sparse),
            "stations_sparse": len(sparse),
            "stations_without_regional_correction": len(no_local),
        },
        "cross_station_speed_spread_mps": consistency.get("spread_mps"),
    }

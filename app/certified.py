"""Hand-off of certified parameters into the localisation dataset.

Only *active* rows of the certification book reach this layer, and only the
localisation pipeline consumes them — calibration evidence and the diagnostic
library always keep the registered station coordinates and the baseline
propagation speed, so an attribution diagnosis can never certify against its
own corrected output.

Three certified parameter types are applied:

* ``propagation_speed`` overrides the propagation speed of the run;
* ``coordinate`` / ``clock_coordinate`` shift the measurement point used for
  distance computations (the station register itself is never rewritten);
* ``clock`` / ``clock_coordinate`` add a residual-level offset/drift correction
  on top of the per-boot clock calibration already applied to arrivals.
"""

from __future__ import annotations

from dataclasses import replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .diagnostics import H_CLOCK, H_COMBINED, H_COORDINATE, H_SPEED
from .models import CertifiedParameter, CertificationRecord


def _active(db: Session) -> list[CertifiedParameter]:
    return list(
        db.execute(
            select(CertifiedParameter).where(CertifiedParameter.active.is_(True))
        ).scalars()
    )


def effective_settings(db: Session, cfg: Settings) -> tuple[Settings, list[int]]:
    """Return run settings with the certified regional speed applied.

    A region-scoped speed wins over a global (region_id IS NULL) one.  The
    returned list records the certified parameter ids consumed, for the run
    snapshot.
    """
    rows = [p for p in _active(db) if p.parameter_type == H_SPEED]
    if not rows:
        return cfg, []
    regional = next((p for p in rows if p.region_id is not None), None)
    chosen = regional or rows[0]
    speed = float(chosen.params_json.get("speed_mps", cfg.propagation_speed_mps))
    return replace(cfg, propagation_speed_mps=speed), [p.id for p in rows]


def coordinate_overrides(db: Session) -> dict[str, dict]:
    """Station measurement-point shifts from active coordinate certifications."""
    out: dict[str, dict] = {}
    for p in _active(db):
        if p.parameter_type not in (H_COORDINATE, H_COMBINED) or not p.station_id:
            continue
        params = p.params_json or {}
        out[p.station_id] = {
            "dx_m": float(params.get("dx_m", 0.0)),
            "dy_m": float(params.get("dy_m", 0.0)),
            "certification_id": p.certification_id,
        }
    return out


def clock_correction_us(
    db: Session, station_id: str, boot_id: str, reference_arrival_us: int
) -> tuple[float, int | None]:
    """Residual-level clock correction at a reference instant.

    Returns ``(correction_us, certification_id)``; subtract the correction from
    the calibrated arrival.  Only one active clock parameter per (station,
    boot) can exist because publishing supersedes the previous version.
    """
    rows = list(
        db.execute(
            select(CertifiedParameter)
            .join(
                CertificationRecord,
                CertificationRecord.id == CertifiedParameter.certification_id,
            )
            .where(
                CertifiedParameter.active.is_(True),
                CertifiedParameter.parameter_type.in_([H_CLOCK, H_COMBINED]),
                CertifiedParameter.station_id == station_id,
                CertifiedParameter.boot_id == boot_id,
            )
        ).scalars()
    )
    if not rows:
        return 0.0, None
    p = rows[0]
    params = p.params_json or {}
    offset = float(params.get("offset_us", 0.0))
    drift = float(params.get("drift_ppm", 0.0))  # us per second
    t0 = float(params.get("t0_us", reference_arrival_us))
    correction = offset + drift * (reference_arrival_us - t0) / 1e6
    return correction, p.certification_id


def certified_summary(db: Session) -> dict:
    """Compact snapshot of what the localisation dataset currently consumes."""
    rows = _active(db)
    return {
        "active_parameter_count": len(rows),
        "parameters": [
            {
                "id": p.id,
                "certification_id": p.certification_id,
                "parameter_type": p.parameter_type,
                "station_id": p.station_id,
                "boot_id": p.boot_id,
                "region_id": p.region_id,
                "version": p.version,
                "params": p.params_json,
                "affected_event_versions": p.affected_event_versions,
            }
            for p in rows
        ],
    }

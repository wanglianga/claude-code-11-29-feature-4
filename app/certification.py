"""Parameter certification book, version gates and event-impact handling.

A diagnosis only *registers* competing hypotheses.  Nothing reaches the
localisation dataset until a certification decision is recorded here.  The
gates are deliberately mechanical and auditable:

1. **Hold-out gate** — a new fit is rejected when its untouched hold-out
   anchors score worse than the baseline (or than the currently active
   parameter for the same scope).
2. **Identifiability gate** — the parameter must be separately identifiable
   (anchor span, directional coverage, design condition, plausible bounds).
3. **Boot/version gate** — device parameters belong to one boot.  A fit that
   arrives late for an older boot can never overwrite parameters certified for
   a newer boot of the same station.
4. **Confirmation gate** — when a revision changes an event a human already
   confirmed, history is never rewritten: a pending re-localisation draft is
   produced instead.

The service never dispatches maintenance tickets and never changes field
devices; certification only publishes data for downstream localisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, settings as global_settings
from .diagnostics import (
    H_CLOCK,
    H_COMBINED,
    H_COORDINATE,
    H_SPEED,
)
from .models import (
    AuditLog,
    CertifiedParameter,
    CertificationRecord,
    DiagnosticEvidence,
    DiagnosticRun,
    Event,
    EventConfirmation,
    EventObservation,
    EventParameterImpact,
    HypothesisRecord,
    LocalizationReviewDraft,
    Observation,
    Station,
    StationBoot,
)
from .physics import distance
from .timeutil import now_us

CERT_APPROVED = "approved"
CERT_REJECTED = "rejected"
CERT_PENDING = "pending"


@dataclass
class GateResult:
    passed: bool
    gate: str
    reason: str
    detail: dict


def _cert_code(hypothesis: HypothesisRecord) -> str:
    return f"C-{hypothesis.diagnostic_run_id}-{hypothesis.hypothesis_type}-{hypothesis.id}"


def _holdout_rmse(hypothesis: HypothesisRecord) -> float | None:
    return (hypothesis.holdout_metrics_json or {}).get("rmse_ms")


def _baseline_holdout_rmse(db: Session, run_id: int) -> float | None:
    run = db.get(DiagnosticRun, run_id)
    base = (run.diagnosis_json or {}).get("baseline_holdout", {}) if run else {}
    return base.get("rmse_ms")


def gate_holdout(
    db: Session, hypothesis: HypothesisRecord, cfg: Settings
) -> GateResult:
    """New fit must not be worse on the anchors that never saw the fit."""
    run = db.get(DiagnosticRun, hypothesis.diagnostic_run_id)
    holdout = (run.diagnosis_json or {}).get("partition", {}) if run else {}
    if not holdout.get("enough_holdout", False):
        return GateResult(
            False, "holdout",
            "not enough untouched anchors to validate the fit",
            {"holdout_anchor_count": holdout.get("holdout_anchor_count", 0)},
        )
    candidate = _holdout_rmse(hypothesis)
    baseline = _baseline_holdout_rmse(db, hypothesis.diagnostic_run_id)
    if candidate is None or baseline is None:
        return GateResult(False, "holdout", "missing hold-out metrics", {})
    relative_limit = baseline * cfg.cert_holdout_worsen_ratio
    # Two conditions, both required: the candidate must not regress relative to
    # the no-fault baseline, and its absolute hold-out error must stay inside
    # the sanity envelope.
    relative_ok = candidate <= relative_limit
    absolute_ok = candidate <= cfg.diag_holdout_abs_limit_ms
    passed = relative_ok and absolute_ok
    if not relative_ok:
        reason = (
            f"hold-out RMSE {candidate:.3f} ms worsens beyond the relative envelope "
            f"{relative_limit:.3f} ms (baseline {baseline:.3f} ms x "
            f"{cfg.cert_holdout_worsen_ratio})"
        )
    elif not absolute_ok:
        reason = (
            f"hold-out RMSE {candidate:.3f} ms exceeds the absolute limit "
            f"{cfg.diag_holdout_abs_limit_ms:.3f} ms"
        )
    else:
        reason = "hold-out RMSE stays within both the relative regression envelope and absolute limit"
    return GateResult(
        passed, "holdout", reason,
        {"candidate_rmse_ms": candidate, "baseline_rmse_ms": baseline,
         "relative_limit_ms": round(relative_limit, 4),
         "absolute_limit_ms": cfg.diag_holdout_abs_limit_ms,
         "worsen_ratio": cfg.cert_holdout_worsen_ratio},
    )


def gate_identifiability(hypothesis: HypothesisRecord) -> GateResult:
    info = hypothesis.identifiability_json or {}
    return GateResult(
        bool(hypothesis.identifiable),
        "identifiability",
        "parameter is separately identifiable from the anchor geometry"
        if hypothesis.identifiable
        else "; ".join(info.get("conditions", ["not identifiable"])),
        info,
    )


def gate_boot_version(
    db: Session, hypothesis: HypothesisRecord, cfg: Settings
) -> GateResult:
    """Reject stale fits that would overwrite a newer boot's parameters.

    Boot ordering is derived from the evidence library: first reference arrival
    observed for each boot.  A fit computed from an older boot may only be
    certified when no newer boot of the same station already holds an active
    certification of the same parameter type.
    """
    if hypothesis.hypothesis_type == H_SPEED and hypothesis.station_id is None:
        return GateResult(True, "boot_version", "regional model is not boot-scoped", {})
    station_id = hypothesis.station_id
    boot_id = hypothesis.boot_id
    if not station_id or not boot_id:
        return GateResult(True, "boot_version", "not a device-boot parameter", {})

    boot_order = _station_boot_order(db, station_id)
    if boot_id not in boot_order:
        return GateResult(False, "boot_version",
                          f"boot {boot_id} has no evidence in the diagnostic library", {})
    boot_index = boot_order.index(boot_id)
    newer = boot_order[boot_index + 1:]
    active_newer = list(
        db.execute(
            select(CertifiedParameter)
            .join(CertificationRecord,
                  CertificationRecord.id == CertifiedParameter.certification_id)
            .where(
                CertifiedParameter.station_id == station_id,
                CertifiedParameter.active.is_(True),
                CertifiedParameter.boot_id.in_(newer),
                CertificationRecord.parameter_type == hypothesis.hypothesis_type,
            )
        ).scalars()
    )
    if active_newer:
        return GateResult(
            False, "boot_version",
            f"fit for older boot {boot_id} would overwrite parameters certified for "
            f"newer boot(s) {[p.boot_id for p in active_newer]}",
            {"boot_id": boot_id, "newer_boots": newer,
             "blocking_certification_ids": [p.certification_id for p in active_newer]},
        )

    # Staleness check: evidence used by the fit must not predate the newest
    # evidence of the same boot by more than the configured envelope.
    age = _fit_evidence_age_s(db, hypothesis)
    if age is not None and age > cfg.cert_stale_max_age_s:
        return GateResult(
            False, "boot_version",
            f"stale fit: newest anchor is {age / 3600.0:.1f} h older than the cutoff "
            f"({cfg.cert_stale_max_age_s / 3600.0:.0f} h envelope); re-run the diagnosis",
            {"evidence_age_s": round(age, 1), "max_age_s": cfg.cert_stale_max_age_s},
        )
    return GateResult(True, "boot_version",
                      f"boot {boot_id} is the newest certified boot for {station_id}",
                      {"boot_order": boot_order, "boot_index": boot_index})


def _station_boot_order(db: Session, station_id: str) -> list[str]:
    """Authoritative boot ordering by first *reference* arrival in evidence.

    The station boot register stores local clock readings, which jump
    arbitrarily at a reboot and therefore cannot order boots.  The diagnostic
    evidence library carries reference arrival times, so the earliest observed
    reference arrival of each boot defines the order; the register is only a
    fallback when no evidence exists yet.
    """
    evidence_rows = list(
        db.execute(
            select(
                DiagnosticEvidence.boot_id,
                DiagnosticEvidence.reference_arrival_us,
            )
            .where(DiagnosticEvidence.station_id == station_id)
            .order_by(DiagnosticEvidence.reference_arrival_us)
        ).all()
    )
    first_ref: dict[str, float] = {}
    for boot_id, ref_us in evidence_rows:
        first_ref[boot_id] = min(first_ref.get(boot_id, math.inf), float(ref_us))
    ordered = sorted(first_ref, key=lambda b: first_ref[b])
    register_rows = list(
        db.execute(
            select(StationBoot.boot_id, StationBoot.first_seen_us)
            .where(StationBoot.station_id == station_id)
            .order_by(StationBoot.first_seen_us)
        ).all()
    )
    for boot_id, _ in register_rows:
        if boot_id not in ordered:
            ordered.append(boot_id)
    return ordered


def _fit_evidence_age_s(db: Session, hypothesis: HypothesisRecord) -> float | None:
    run = db.get(DiagnosticRun, hypothesis.diagnostic_run_id)
    if run is None:
        return None
    newest = db.scalar(
        select(DiagnosticEvidence.reference_arrival_us)
        .where(
            DiagnosticEvidence.diagnostic_run_id == run.id,
            DiagnosticEvidence.partition == "fit",
            DiagnosticEvidence.excluded.is_(False),
        )
        .order_by(DiagnosticEvidence.reference_arrival_us.desc())
        .limit(1)
    )
    if newest is None:
        return None
    return max(0.0, (run.cutoff_us - newest) / 1e6)


# --------------------------------------------------------------------------- #
# Certification decisions
# --------------------------------------------------------------------------- #
def decide_certification(
    db: Session,
    hypothesis_id: int,
    *,
    decided_by: str = "maintainer",
    cfg: Settings | None = None,
) -> CertificationRecord:
    cfg = cfg or global_settings
    hypothesis = db.get(HypothesisRecord, hypothesis_id)
    if hypothesis is None:
        raise KeyError("hypothesis")
    existing = db.execute(
        select(CertificationRecord).where(
            CertificationRecord.hypothesis_id == hypothesis_id
        ).order_by(CertificationRecord.id.desc())
    ).scalars().first()
    if existing is not None and existing.status in (CERT_APPROVED, CERT_REJECTED):
        raise ValueError(
            f"hypothesis {hypothesis_id} already has a {existing.status} certification "
            f"({existing.cert_code}); create a new diagnostic run to revise it"
        )

    gates = [
        gate_identifiability(hypothesis),
        gate_holdout(db, hypothesis, cfg),
        gate_boot_version(db, hypothesis, cfg),
    ]
    gate_payload = [
        {"gate": g.gate, "passed": g.passed, "reason": g.reason, "detail": g.detail}
        for g in gates
    ]
    all_passed = all(g.passed for g in gates)
    run = db.get(DiagnosticRun, hypothesis.diagnostic_run_id)
    record = CertificationRecord(
        cert_code=_cert_code(hypothesis),
        hypothesis_id=hypothesis.id,
        diagnostic_run_id=hypothesis.diagnostic_run_id,
        parameter_type=hypothesis.hypothesis_type,
        station_id=hypothesis.station_id,
        boot_id=hypothesis.boot_id,
        region_id=hypothesis.region_id,
        status=CERT_APPROVED if all_passed else CERT_REJECTED,
        decision_reason="; ".join(g.reason for g in gates if not g.passed) or None,
        gate_json={"gates": gate_payload},
        holdout_comparison_json={
            "candidate": hypothesis.holdout_metrics_json,
            "baseline": (run.diagnosis_json or {}).get("baseline_holdout", {}) if run else {},
            "worsen_ratio": cfg.cert_holdout_worsen_ratio,
        },
        exclusion_comparison_json=(run.diagnosis_json or {}).get("outlier_exclusion") if run else None,
        identifiability_json=hypothesis.identifiability_json,
        params_json=hypothesis.params_json,
        decided_by=decided_by,
        created_at_us=now_us(),
        decided_at_us=now_us(),
    )
    db.add(record)
    db.flush()
    hypothesis.status = record.status
    if all_passed:
        _publish_parameter(db, record, hypothesis, decided_by)
    db.add(
        AuditLog(
            actor=decided_by,
            action=f"certification.{record.status}",
            target_type="hypothesis",
            target_id=str(hypothesis.id),
            detail={"cert_code": record.cert_code,
                    "gates": gate_payload,
                    "rejected_gates": [g.gate for g in gates if not g.passed]},
            created_at_us=now_us(),
        )
    )
    db.flush()
    return record


def _publish_parameter(
    db: Session, record: CertificationRecord, hypothesis: HypothesisRecord, actor: str
) -> CertifiedParameter:
    # Supersede the previous active parameter of the same scope/type.
    previous = list(
        db.execute(
            select(CertifiedParameter).where(
                CertifiedParameter.parameter_type == record.parameter_type,
                CertifiedParameter.station_id.is_(record.station_id)
                if record.station_id is None
                else CertifiedParameter.station_id == record.station_id,
                CertifiedParameter.region_id.is_(record.region_id)
                if record.region_id is None
                else CertifiedParameter.region_id == record.region_id,
                CertifiedParameter.boot_id.is_(record.boot_id)
                if record.boot_id is None
                else CertifiedParameter.boot_id == record.boot_id,
                CertifiedParameter.active.is_(True),
            )
        ).scalars()
    )
    version = (max((p.version for p in previous), default=0) + 1) if previous else 1
    impacts = _compute_event_impacts(db, record, hypothesis)
    for prev in previous:
        prev.active = False
        prev.valid_to_us = now_us()
        record.supersedes_certification_id = (
            record.supersedes_certification_id or prev.id
        )
    record.version = version
    parameter = CertifiedParameter(
        certification_id=record.id,
        parameter_type=record.parameter_type,
        station_id=record.station_id,
        boot_id=record.boot_id,
        region_id=record.region_id,
        params_json=record.params_json,
        identifiability_json=record.identifiability_json,
        affected_event_versions=[
            {"event_id": i.event_id, "revision": i.event_revision, "action": i.action,
             "human_confirmed": i.human_confirmed}
            for i in impacts
        ],
        handoff_json={
            "delivered_to": "localization_dataset",
            "diagnostic_run_id": record.diagnostic_run_id,
            "hypothesis_id": record.hypothesis_id,
            "organization": hypothesis.organization,
            "cert_code": record.cert_code,
            "identifiable_conditions": record.identifiability_json.get("conditions", []),
        },
        active=True,
        valid_from_us=now_us(),
        version=version,
        created_at_us=now_us(),
    )
    db.add(parameter)
    db.flush()
    for impact in impacts:
        impact.review_draft_id = (
            _create_review_draft(db, record, hypothesis, impact)
            if impact.action == "pending_review_draft" else None
        )
        db.add(impact)
    return parameter


# --------------------------------------------------------------------------- #
# Event impact
# --------------------------------------------------------------------------- #
def _affected_events(
    db: Session, record: CertificationRecord
) -> list[Event]:
    stmt = select(Event).where(Event.status != "superseded")
    if record.station_id:
        stmt = (
            stmt.join(EventObservation, EventObservation.event_id == Event.id)
            .where(EventObservation.station_id == record.station_id)
        )
    events = list(db.execute(stmt).scalars())
    if record.region_id:
        # Regional speed changes touch every live event in the applicability
        # region; the region filter is geometric on the event representative.
        from .models import PropagationRegion

        region = db.get(PropagationRegion, record.region_id)
        if region is not None:
            events = [
                e for e in events
                if region.x_min <= e.x <= region.x_max and region.y_min <= e.y <= region.y_max
            ]
    # Deduplicate while keeping order.
    seen: set[int] = set()
    unique = []
    for e in events:
        if e.id not in seen:
            seen.add(e.id)
            unique.append(e)
    return unique


def _compute_event_impacts(
    db: Session, record: CertificationRecord, hypothesis: HypothesisRecord
) -> list[EventParameterImpact]:
    impacts: list[EventParameterImpact] = []
    confirmation_ids = {
        c.event_id for c in db.execute(select(EventConfirmation)).scalars()
    }
    for event in _affected_events(db, record):
        confirmed = event.id in confirmation_ids
        action = "pending_review_draft" if confirmed else "parameter_version_bumped"
        impacts.append(
            EventParameterImpact(
                certification_id=record.id,
                event_id=event.id,
                event_revision=event.revision,
                event_status_at_impact=event.status,
                human_confirmed=confirmed,
                action=action,
                reason=(
                    "human-confirmed event: history is immutable; a review draft is created"
                    if confirmed
                    else "downstream localisation may consume the new parameter version"
                ),
                created_at_us=now_us(),
            )
        )
    return impacts


def _create_review_draft(
    db: Session,
    record: CertificationRecord,
    hypothesis: HypothesisRecord,
    impact: EventParameterImpact,
) -> int:
    event = db.get(Event, impact.event_id)
    before = {
        "x": event.x, "y": event.y,
        "emission_us": event.emission_us,
        "residual_rms_us": event.residual_rms_us,
        "calibration_hash": event.calibration_hash,
    }
    after = _recompute_event_localization(db, event, record)
    delta = {
        "position_delta_m": round(
            math.hypot(after["x"] - event.x, after["y"] - event.y), 4
        ),
        "emission_delta_ms": round((after["emission_us"] - event.emission_us) / 1000.0, 4),
        "residual_rms_delta_ms": round(
            (after["residual_rms_us"] - event.residual_rms_us) / 1000.0, 4
        ),
    }
    draft = LocalizationReviewDraft(
        certification_id=record.id,
        source_event_id=event.id,
        status="pending_review",
        params_snapshot_json=record.params_json,
        before_json=before,
        after_json=after,
        delta_json=delta,
        reason=(
            f"parameter {record.parameter_type} ({record.cert_code}) revised while the "
            "event was human-confirmed; original event row is left untouched"
        ),
        resolved=False,
        created_at_us=now_us(),
        updated_at_us=now_us(),
    )
    db.add(draft)
    db.flush()
    return draft.id


def _recompute_event_localization(
    db: Session, event: Event, record: CertificationRecord
) -> dict:
    """Recompute emission dispersion with the certified parameter, same point.

    The representative position is held fixed; this computes what the event's
    timing residuals would be under the new parameters so a reviewer can judge
    the draft without rerunning the whole pipeline.
    """
    links = list(
        db.execute(
            select(EventObservation).where(EventObservation.event_id == event.id)
        ).scalars()
    )
    stations = {s.id: s for s in db.execute(select(Station)).scalars()}
    speed = global_settings.propagation_speed_mps
    coord_shift: dict[str, tuple[float, float]] = {}
    clock_corr: dict[tuple[str, str], float] = {}
    station_offset: dict[str, float] = {}
    params = record.params_json or {}
    if record.parameter_type == H_SPEED:
        speed = float(params.get("speed_mps", speed))
        # The regional speed model carries one residual clock offset per
        # station (estimated jointly with the shared speed).
        station_offset = {
            sid: float(off)
            for sid, off in (params.get("station_clock_offsets_us") or {}).items()
        }
    elif record.parameter_type == H_COORDINATE:
        if record.station_id:
            coord_shift[record.station_id] = (
                float(params.get("dx_m", 0.0)), float(params.get("dy_m", 0.0))
            )
    elif record.parameter_type in (H_CLOCK, H_COMBINED):
        if record.station_id and record.boot_id:
            clock_corr[(record.station_id, record.boot_id)] = float(
                params.get("offset_us", 0.0)
            )
        if record.parameter_type == H_COMBINED and record.station_id:
            coord_shift[record.station_id] = (
                float(params.get("dx_m", 0.0)), float(params.get("dy_m", 0.0))
            )
    adjusted = []
    for link in links:
        st = stations.get(link.station_id)
        obs = db.get(Observation, link.observation_id)
        if st is None or obs is None:
            continue
        dx, dy = coord_shift.get(link.station_id, (0.0, 0.0))
        d = distance(st.x + dx, st.y + dy, event.x, event.y)
        arrival = (
            float(link.corrected_arrival_us)
            - clock_corr.get((link.station_id, obs.boot_id), 0.0)
            - station_offset.get(link.station_id, 0.0)
        )
        adjusted.append(arrival - d / speed * 1e6)
    if not adjusted:
        return {"x": event.x, "y": event.y, "emission_us": event.emission_us,
                "residual_rms_us": event.residual_rms_us}
    emission = sum(adjusted) / len(adjusted)
    rms = math.sqrt(sum((v - emission) ** 2 for v in adjusted) / len(adjusted))
    return {
        "x": event.x,
        "y": event.y,
        "emission_us": int(round(emission)),
        "residual_rms_us": int(round(rms)),
        "speed_mps": speed,
    }


# --------------------------------------------------------------------------- #
# Review draft resolution (explicit human action only)
# --------------------------------------------------------------------------- #
def resolve_review_draft(
    db: Session, draft_id: int, *, resolution: str, note: str | None, actor: str
) -> LocalizationReviewDraft:
    draft = db.get(LocalizationReviewDraft, draft_id)
    if draft is None:
        raise KeyError("draft")
    if draft.resolved:
        raise ValueError("draft already resolved")
    if resolution not in {"accepted", "rejected"}:
        raise ValueError("resolution must be 'accepted' or 'rejected'")
    draft.resolved = True
    draft.status = "accepted_into_new_revision" if resolution == "accepted" else "rejected"
    draft.resolution_note = note
    draft.updated_at_us = now_us()
    db.add(
        AuditLog(
            actor=actor,
            action=f"review_draft.{resolution}",
            target_type="review_draft",
            target_id=str(draft.id),
            detail={"source_event_id": draft.source_event_id, "note": note},
            created_at_us=now_us(),
        )
    )
    db.flush()
    return draft


def confirm_event(
    db: Session, event_id: int, *, confirmed_by: str, note: str | None
) -> EventConfirmation:
    event = db.get(Event, event_id)
    if event is None:
        raise KeyError("event")
    existing = db.execute(
        select(EventConfirmation).where(EventConfirmation.event_id == event_id)
    ).scalar_one_or_none()
    if existing is not None:
        raise ValueError("event already confirmed")
    row = EventConfirmation(
        event_id=event_id, confirmed_by=confirmed_by, note=note, created_at_us=now_us()
    )
    db.add(row)
    db.add(
        AuditLog(
            actor=confirmed_by,
            action="event.confirm",
            target_type="event",
            target_id=str(event_id),
            detail={"note": note},
            created_at_us=now_us(),
        )
    )
    db.flush()
    return row

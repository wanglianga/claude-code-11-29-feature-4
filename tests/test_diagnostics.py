"""Unit and integration tests for calibration fault-attribution diagnostics.

The pure-fitter tests build synthetic anchors directly (no database) and cover
the three fault signatures: a temporal clock step, a direction-reversing
coordinate error and a distance-correlated propagation-speed change.  The
integration tests drive the full API over the bundled ``diagnosis`` and
``sparse_anchors`` simulator scenarios.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app import diagnostics as dg
from app.config import Settings

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SPEED = 343.0
T0 = 1_794_600_000_000_000


def _cfg(**over) -> Settings:
    base = Settings()
    return base.__class__(**{**base.__dict__, **over})


def _anchor(
    i: int,
    *,
    station: str = "S1",
    ref_time_us: int = T0,
    residual_us: float = 0.0,
    distance_m: float = 500.0,
    direction_rad: float = 0.0,
    sx: float = 0.0,
    sy: float = 0.0,
) -> dg.DiagAnchor:
    return dg.DiagAnchor(
        evidence_id=1000 + i,
        observation_id=2000 + i,
        station_id=station,
        boot_id="b1",
        source_id=f"src{i}",
        source_x=sx,
        source_y=sy,
        station_x=sx + math.cos(direction_rad) * distance_m,
        station_y=sy + math.sin(direction_rad) * distance_m,
        emission_us=ref_time_us,
        distance_m=distance_m,
        direction_rad=direction_rad,
        expected_arrival_us=ref_time_us,
        measured_arrival_us=ref_time_us + residual_us,
        residual_us=residual_us,
        local_capture_us=ref_time_us,
    )


def _ring_anchors(n: int = 16, residual_fn=None) -> list[dg.DiagAnchor]:
    """Anchors surrounding a station at evenly spaced directions."""
    out = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        residual = residual_fn(angle, i) if residual_fn else 0.0
        out.append(
            _anchor(
                i,
                ref_time_us=T0 + i * 120_000_000,
                residual_us=residual,
                distance_m=400.0 + 100.0 * (i % 4),
                direction_rad=angle,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Hold-out partition
# --------------------------------------------------------------------------- #
def test_holdout_partition_is_deterministic_and_stratified():
    cfg = _cfg()
    anchors = [
        _anchor(i, ref_time_us=T0 + i * 60_000_000, direction_rad=i * 0.5)
        for i in range(24)
    ]
    fit1, hold1 = dg.partition_anchors(anchors, cfg)
    fit2, hold2 = dg.partition_anchors(anchors, cfg)
    ids1 = {a.evidence_id for a in hold1}
    ids2 = {a.evidence_id for a in hold2}
    assert ids1 == ids2  # deterministic
    assert len(hold1) >= cfg.diag_min_holdout
    assert not ids1.intersection(a.evidence_id for a in fit1)
    assert len(fit1) + len(hold1) == len(anchors)


def test_sparse_anchors_cannot_form_a_holdout():
    cfg = _cfg()
    anchors = [_anchor(i, ref_time_us=T0 + i * 60_000_000) for i in range(4)]
    fit, holdout = dg.partition_anchors(anchors, cfg)
    assert holdout == []
    assert len(fit) == 4


# --------------------------------------------------------------------------- #
# Directional analysis
# --------------------------------------------------------------------------- #
def test_same_sign_residuals_flag_clock_offset_plausibility():
    cfg = _cfg()
    anchors = _ring_anchors(residual_fn=lambda angle, i: 80_000.0)
    analysis = dg.direction_analysis(anchors, cfg)
    assert analysis["clock_offset_plausible"] is True
    assert analysis["single_clock_fix_inadequate"] is False


def test_direction_reversal_flags_single_clock_fix_inadequate():
    cfg = _cfg()
    # Residual flips sign across the x-axis: the geometric signature of a
    # station registered at the wrong y coordinate.
    anchors = _ring_anchors(
        residual_fn=lambda angle, i: 80_000.0 * (1.0 if math.sin(angle) >= 0 else -1.0)
    )
    analysis = dg.direction_analysis(anchors, cfg)
    assert analysis["direction_reversal"] is True
    assert analysis["single_clock_fix_inadequate"] is True


# --------------------------------------------------------------------------- #
# Clock hypothesis (offset/drift + step)
# --------------------------------------------------------------------------- #
def test_clock_step_is_detected_and_sized():
    cfg = _cfg()
    anchors = []
    for i in range(40):
        step = 180_000.0 if i >= 30 else 0.0
        anchors.append(
            _anchor(
                i,
                ref_time_us=T0 + i * 120_000_000,
                residual_us=step,
                distance_m=400.0 + 50.0 * (i % 3),
                direction_rad=(i % 8) * math.pi / 4,
            )
        )
    hyp = dg.fit_clock(anchors, dg.ORG_PER_STATION, cfg, "S1", "b1")
    assert hyp.identifiable
    assert hyp.params["model"] == "affine_with_step"
    assert hyp.params["step_ms"] == pytest.approx(180.0, abs=5.0)
    # The prediction models the residual; on the late side it must reproduce the
    # 180 ms step rather than the pre-step level.
    assert hyp.predict(anchors[35]) == pytest.approx(180_000.0, abs=5_000.0)
    assert abs(hyp.predict(anchors[5])) < 5_000.0


def test_pure_affine_clock_when_no_step_present():
    cfg = _cfg()
    anchors = [
        _anchor(
            i,
            ref_time_us=T0 + i * 120_000_000,
            residual_us=2_000.0 + 5.0 * i,
            direction_rad=(i % 8) * math.pi / 4,
        )
        for i in range(30)
    ]
    hyp = dg.fit_clock(anchors, dg.ORG_PER_STATION, cfg, "S1", "b1")
    assert hyp.params["model"] == "affine"
    assert hyp.fit_metrics["step_detected"] is False


# --------------------------------------------------------------------------- #
# Coordinate hypothesis
# --------------------------------------------------------------------------- #
def test_coordinate_shift_recovers_known_displacement():
    cfg = _cfg()
    dx, dy = 60.0, -40.0
    anchors = []
    for i in range(24):
        angle = 2 * math.pi * i / 24
        ux, uy = math.cos(angle), math.sin(angle)
        # A shifted station adds (u . shift) / speed to the arrival residual.
        residual = (ux * dx + uy * dy) / SPEED * 1e6
        anchors.append(
            _anchor(
                i,
                ref_time_us=T0 + i * 90_000_000,
                residual_us=residual,
                distance_m=300.0 + 60.0 * i,  # wide spread keeps the shift identifiable
                direction_rad=angle,
            )
        )
    hyp = dg.fit_coordinate(anchors, SPEED, dg.ORG_PER_STATION, cfg, "S2", "b1")
    assert hyp.identifiable, hyp.identifiability
    assert hyp.params["dx_m"] == pytest.approx(dx, abs=8.0)
    assert hyp.params["dy_m"] == pytest.approx(dy, abs=8.0)
    assert hyp.params["shift_m"] == pytest.approx(math.hypot(dx, dy), abs=8.0)


# --------------------------------------------------------------------------- #
# Propagation-speed hypothesis
# --------------------------------------------------------------------------- #
def test_speed_hypothesis_recovers_regional_speed_change():
    cfg = _cfg()
    true_speed = 328.0
    anchors = []
    for i in range(24):
        distance_m = 300.0 + 90.0 * i  # wide distance spread
        angle = (i % 8) * math.pi / 4
        residual = distance_m * (1.0 / true_speed - 1.0 / SPEED) * 1e6
        anchors.append(
            _anchor(
                i,
                ref_time_us=T0 + i * 90_000_000,
                residual_us=residual,
                distance_m=distance_m,
                direction_rad=angle,
            )
        )
    hyp = dg.fit_speed(
        anchors, SPEED, cfg, organization=dg.ORG_PER_STATION,
        station_id="S1", boot_id="b1",
    )
    assert hyp.identifiable, hyp.identifiability
    assert hyp.params["speed_mps"] == pytest.approx(true_speed, abs=2.0)


def test_json_safe_replaces_non_finite_values():
    payload = {"a": float("inf"), "b": [float("nan"), 1.0], "c": {"d": float("-inf")}}
    safe = dg._json_safe(payload)
    import json

    # Must round-trip through strict JSON (JSONB would reject Infinity/NaN).
    parsed = json.loads(json.dumps(safe))
    assert parsed["a"] is None
    assert parsed["b"][0] is None
    assert parsed["c"]["d"] is None
    assert parsed["b"][1] == 1.0


# --------------------------------------------------------------------------- #
# Integration over the bundled fault scenario
# --------------------------------------------------------------------------- #
def _region(client):
    res = client.post(
        "/propagation-regions",
        json={
            "id": "R1",
            "name": "north",
            "x_min": -400,
            "x_max": 2300,
            "y_min": -700,
            "y_max": 1500,
            "baseline_speed_mps": 343.0,
        },
    )
    assert res.status_code == 201, res.text


def _station_summary(client, station: str) -> dict:
    run = client.post(
        "/diagnostic-runs",
        json={"scope_type": "station", "station_id": station},
    ).json()
    assert run["status"] == "completed"
    summaries = run["diagnosis"]["organizations"]["per_station"]["stations"]
    return summaries[0]


def test_diagnosis_scenario_attributes_clock_step_to_s1(client, api, tmp_path):
    from simulator.run_simulator import load_scenario

    load_scenario("diagnosis", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    _region(client)
    summary = _station_summary(client, "S1")
    clock = next(h for h in summary["hypotheses"] if h["type"] == "clock")
    assert clock["rank"] == 1
    detail = client.get(f"/hypotheses/{clock['hypothesis_id']}").json()
    assert detail["params"]["model"] == "affine_with_step"
    assert detail["params"]["step_ms"] == pytest.approx(180.0, abs=30.0)
    # A direction-reversing geometry fix must not be preferred.
    assert "clock step" in clock["reason"].lower()


def test_diagnosis_scenario_recovers_s2_coordinate_shift(client, api, tmp_path):
    from simulator.run_simulator import load_scenario

    load_scenario("diagnosis", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    _region(client)
    summary = _station_summary(client, "S2")
    coord = next(h for h in summary["hypotheses"] if h["type"] == "coordinate")
    assert coord["rank"] == 1
    detail = client.get(f"/hypotheses/{coord['hypothesis_id']}").json()
    # True registered error is 80 m west (dx = +80 m correction).
    assert detail["params"]["dx_m"] == pytest.approx(80.0, abs=25.0)
    assert detail["params"]["shift_m"] >= 50.0


def test_diagnosis_scenario_excludes_outlier_with_before_after(client, api, tmp_path):
    from simulator.run_simulator import load_scenario

    load_scenario("diagnosis", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    _region(client)
    run = client.post(
        "/diagnostic-runs", json={"scope_type": "station", "station_id": "S3"}
    ).json()
    exclusion = run["diagnosis"]["outlier_exclusion"]
    assert exclusion["excluded_count"] >= 1
    # The before/after comparison must be present and anchored on the untouched
    # hold-out partition.
    assert "holdout_before_exclusion" in exclusion
    assert exclusion["fit_anchor_count_after"] < exclusion["fit_anchor_count_before"]
    excluded = client.get(
        f"/diagnostic-runs/{run['id']}/evidence", params={"excluded": True}
    ).json()
    assert excluded and all(e["excluded"] for e in excluded)
    assert all(e["exclusion_reason"] for e in excluded)


def test_regional_organisation_recovers_shared_speed(client, api, tmp_path):
    from simulator.run_simulator import load_scenario

    load_scenario("diagnosis", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    _region(client)
    run = client.post(
        "/diagnostic-runs", json={"scope_type": "region", "region_id": "R1"}
    ).json()
    assert run["status"] == "completed"
    reg = run["diagnosis"]["organizations"]["regional"]
    assert reg["speed_identifiable"] is True
    # Ground-truth speed is 328 m/s against a 343 baseline.
    assert 318.0 <= reg["speed"]["speed_mps"] <= 338.0
    # Both organisations validate the same localisation events.
    validated = run["diagnosis"]["event_validation"]
    assert validated and all(
        {"baseline", "per_station", "regional"} <= set(row) for row in validated
    )
    # Per-station standalone speed estimates are inconsistent at the
    # coordinate-faulted station, exposing device-parameter absorption.
    consistency = run["diagnosis"]["cross_station_speed_consistency"]
    estimates = {e["station_id"]: e["speed_mps"] for e in consistency["per_station_speed_estimates"]}
    assert abs(estimates["S2"] - 328.0) > 10.0


def test_sparse_anchors_leave_every_hypothesis_non_certifiable(client, api, tmp_path):
    from simulator.run_simulator import load_scenario

    load_scenario("sparse_anchors", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    client.post(
        "/propagation-regions",
        json={
            "id": "R1", "name": "n",
            "x_min": -300, "x_max": 2100, "y_min": -500, "y_max": 1300,
            "baseline_speed_mps": 343.0,
        },
    )
    run = client.post(
        "/diagnostic-runs", json={"scope_type": "station", "station_id": "S1"}
    ).json()
    assert run["diagnosis"]["partition"]["enough_holdout"] is False
    for hyp in run["diagnosis"]["organizations"]["per_station"]["stations"][0]["hypotheses"]:
        decision = client.post(
            f"/hypotheses/{hyp['hypothesis_id']}/certify", json={}
        ).json()
        assert decision["status"] == "rejected"
        assert any(g["gate"] == "holdout" and not g["passed"] for g in decision["gates"])
    assert client.get("/certified-parameters").json() == []


def test_station_without_evidence_completes_and_recommends_none(client, api):
    # A station that never uploaded a calibration anchor must not crash the
    # diagnosis, must produce no hypotheses, and must not be pushed toward a
    # parameter organisation it cannot support.
    res = client.post(
        "/stations",
        json={"id": "SEMPTY", "name": "No anchors", "x": 100.0, "y": 100.0},
    )
    assert res.status_code == 201
    run = client.post(
        "/diagnostic-runs", json={"scope_type": "station", "station_id": "SEMPTY"}
    ).json()
    assert run["status"] == "completed"
    assert run["fit_anchor_count"] == 0
    assert run["holdout_anchor_count"] == 0
    assert run["diagnosis"]["organizations"]["per_station"]["stations"] == []
    assert (
        run["diagnosis"]["organization_recommendation"]["recommended"] == "none"
    )
    assert client.get("/hypotheses").json() == []
    assert client.get(
        f"/diagnostic-runs/{run['id']}/evidence"
    ).json() == []

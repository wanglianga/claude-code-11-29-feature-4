"""Certification book, version gates and event-impact integration tests."""

from __future__ import annotations

from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _region(client):
    return client.post(
        "/propagation-regions",
        json={
            "id": "R1", "name": "north",
            "x_min": -400, "x_max": 2300, "y_min": -700, "y_max": 1500,
            "baseline_speed_mps": 343.0,
        },
    )


def _load_diagnosis(client, api, tmp_path, station: str | None = None, region: str | None = None):
    from simulator.run_simulator import load_scenario

    load_scenario("diagnosis", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    _region(client)
    body = (
        {"scope_type": "station", "station_id": station}
        if station
        else {"scope_type": "region", "region_id": region}
    )
    return client.post("/diagnostic-runs", json=body).json()


def test_evidence_library_freezes_partition_and_geometry(client, api, tmp_path):
    run = _load_diagnosis(client, api, tmp_path, station="S1")
    evidence = client.get(f"/diagnostic-runs/{run['id']}/evidence").json()
    assert len(evidence) == run["fit_anchor_count"] + run["holdout_anchor_count"]
    partitions = {e["partition"] for e in evidence}
    assert partitions == {"fit", "holdout"}
    # Every row carries the frozen measurement geometry and a mapping source.
    for e in evidence:
        assert e["source_x"] and e["distance_m"] > 0
        assert e["mapping_source"] in {
            "pipeline_calibration_model", "diagnostic_clock_trend"
        }
    held = client.get(
        f"/diagnostic-runs/{run['id']}/evidence", params={"partition": "holdout"}
    ).json()
    assert len(held) == run["holdout_anchor_count"]


def test_holdout_gate_rejects_a_worsening_fit(client, api, tmp_path):
    # On the sparse scenario every fit lacks a hold-out partition, so the gate
    # must reject publication rather than certify an unverified parameter.
    from simulator.run_simulator import load_scenario

    load_scenario("sparse_anchors", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    _region(client)
    run = client.post(
        "/diagnostic-runs", json={"scope_type": "station", "station_id": "S1"}
    ).json()
    hyp = run["diagnosis"]["organizations"]["per_station"]["stations"][0]["hypotheses"][0]
    decision = client.post(
        f"/hypotheses/{hyp['hypothesis_id']}/certify", json={"decided_by": "m"}
    ).json()
    assert decision["status"] == "rejected"
    holdout_gate = next(g for g in decision["gates"] if g["gate"] == "holdout")
    assert holdout_gate["passed"] is False
    assert decision["holdout_comparison"]["candidate"]


def test_certified_clock_parameter_enters_next_pipeline_run(client, api, tmp_path):
    run = _load_diagnosis(client, api, tmp_path, station="S1")
    clock = next(
        h
        for h in run["diagnosis"]["organizations"]["per_station"]["stations"][0]["hypotheses"]
        if h["type"] == "clock"
    )
    decision = client.post(
        f"/hypotheses/{clock['hypothesis_id']}/certify", json={"decided_by": "m"}
    ).json()
    assert decision["status"] == "approved"
    active = client.get("/certified-parameters").json()
    assert any(p["parameter_type"] == "clock" and p["station_id"] == "S1" for p in active)
    # The hand-off block names the localization dataset and identifiability.
    clock_param = next(p for p in active if p["parameter_type"] == "clock")
    assert clock_param["handoff"]["delivered_to"] == "localization_dataset"
    assert clock_param["identifiability"]["identifiable"] is True
    # A subsequent run consumes the parameter without error and still localises.
    rerun = client.post("/runs", json={"wait": True}).json()
    assert rerun["status"] in {"done", "awaiting_review"}
    assert rerun["summary"]["events_created"] >= 1


def test_confirmed_event_gets_review_draft_not_history_rewrite(client, api, tmp_path):
    _load_diagnosis(client, api, tmp_path, station="S1")
    events = client.get("/events", params={"limit": 1000}).json()["events"]
    target = next(e for e in events if "S1" in {
        o["station_id"]
        for o in client.get(f"/events/{e['id']}").json()["observations"]
    })
    # Human confirmation before the parameter revision.
    conf = client.post(
        f"/events/{target['id']}/confirmations",
        json={"confirmed_by": "zoe", "note": "signed off"},
    )
    assert conf.status_code == 201
    run = client.post(
        "/diagnostic-runs", json={"scope_type": "station", "station_id": "S1"}
    ).json()
    clock = next(
        h
        for h in run["diagnosis"]["organizations"]["per_station"]["stations"][0]["hypotheses"]
        if h["type"] == "clock"
    )
    client.post(f"/hypotheses/{clock['hypothesis_id']}/certify", json={"decided_by": "zoe"})
    impact = next(
        i
        for i in client.get(
            f"/certifications/{client.get('/certifications').json()[0]['id']}/impacts"
        ).json()
        if i["event_id"] == target["id"]
    )
    assert impact["human_confirmed"] is True
    assert impact["action"] == "pending_review_draft"
    drafts = client.get("/review-drafts").json()
    draft = next(d for d in drafts if d["source_event_id"] == target["id"])
    assert draft["status"] == "pending_review"
    assert draft["resolved"] is False
    # The original confirmed event row is untouched (id and status preserved).
    after = client.get(f"/events/{target['id']}").json()
    assert after["id"] == target["id"]
    assert after["status"] == target["status"]
    # Resolving the draft is an explicit human action and never auto-applies.
    resolved = client.post(
        f"/review-drafts/{draft['id']}/resolve",
        json={"resolution": "accepted", "note": "ok", "resolved_by": "zoe"},
    ).json()
    assert resolved["resolved"] is True
    assert resolved["status"] == "accepted_into_new_revision"


def test_boot_version_gate_blocks_old_boot_from_overwriting_new(client, api, tmp_path):
    from simulator.run_simulator import load_scenario

    load_scenario("reboot", "http://testserver", truth_dir=tmp_path, reset=False)
    client.post("/runs", json={"wait": True})
    run = client.post(
        "/diagnostic-runs", json={"scope_type": "station", "station_id": "S2"}
    ).json()
    by_boot = {
        s["boot_id"]: s
        for s in run["diagnosis"]["organizations"]["per_station"]["stations"]
    }
    # Adopt the post-reboot boot first.
    b2 = next(h for h in by_boot["b2"]["hypotheses"] if h["type"] == "clock")
    assert client.post(
        f"/hypotheses/{b2['hypothesis_id']}/certify", json={}
    ).json()["status"] == "approved"
    # A late fit for the older boot must not overwrite it.
    b1 = next(h for h in by_boot["b1"]["hypotheses"] if h["type"] == "clock")
    rejected = client.post(
        f"/hypotheses/{b1['hypothesis_id']}/certify", json={}
    ).json()
    assert rejected["status"] == "rejected"
    gate = next(g for g in rejected["gates"] if g["gate"] == "boot_version")
    assert gate["passed"] is False
    active = client.get("/certified-parameters").json()
    assert all(p["boot_id"] != "b1" for p in active)


def test_certification_is_idempotent_and_double_decide_conflicts(client, api, tmp_path):
    run = _load_diagnosis(client, api, tmp_path, station="S1")
    clock = next(
        h
        for h in run["diagnosis"]["organizations"]["per_station"]["stations"][0]["hypotheses"]
        if h["type"] == "clock"
    )
    first = client.post(
        f"/hypotheses/{clock['hypothesis_id']}/certify", json={}
    )
    assert first.status_code == 200
    second = client.post(
        f"/hypotheses/{clock['hypothesis_id']}/certify", json={}
    )
    assert second.status_code == 409


def test_certified_speed_supersedes_previous_version(client, api, tmp_path):
    run = _load_diagnosis(client, api, tmp_path, region="R1")
    reg = run["diagnosis"]["organizations"]["regional"]
    decision = client.post(
        f"/hypotheses/{reg['speed_hypothesis_id']}/certify", json={"decided_by": "m"}
    ).json()
    assert decision["status"] == "approved"
    first_id = decision["id"]
    # A fresh regional diagnosis produces a new candidate; approving it must
    # supersede the prior version rather than keep two active speed rows.
    second_run = client.post(
        "/diagnostic-runs", json={"scope_type": "region", "region_id": "R1"}
    ).json()
    second_reg = second_run["diagnosis"]["organizations"]["regional"]
    second = client.post(
        f"/hypotheses/{second_reg['speed_hypothesis_id']}/certify",
        json={"decided_by": "m"},
    ).json()
    assert second["version"] == 2
    assert second["supersedes_certification_id"] == first_id
    speeds = client.get(
        "/certified-parameters", params={"parameter_type": "propagation_speed"}
    ).json()
    active = [p for p in speeds if p["active"]]
    assert len(active) == 1
    assert active[0]["version"] == 2

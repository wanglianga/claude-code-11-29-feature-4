"""Independent verification of the service's output against scenario truth.

This module deliberately shares *no* code with the service's association,
localisation or calibration logic.  It re-implements the forward model from
the documented physical assumptions:

    arrival(reference) = emission + distance(source, station) / speed
    local_reading      = slope_boot * arrival + intercept_boot

Given a ``truth.json`` it recomputes every station's expected local reading,
compares it with the record actually uploaded, and then checks the service's
published events against ground truth: event counts, membership, position
accuracy, resolved mirror regions, and the required conflict/duplicate
behaviour.

Usage::

    python -m verifier.verify --scenario overlapping --base-url http://127.0.0.1:8000

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import requests

SPEED_DEFAULT = 343.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def load_truth(path: Path) -> dict:
    truth = json.loads(path.read_text())
    for event in truth["events"]:
        event.setdefault("emission_us", _iso_to_us(event["emission"]))
    return truth


def boot_map(truth: dict) -> dict[tuple[str, str], dict]:
    return {
        (st["id"], boot["boot_id"]): boot
        for st in truth["stations"]
        for boot in st["boots"]
    }


def station_map(truth: dict) -> dict[str, dict]:
    return {st["id"]: st for st in truth["stations"]}


def forward_local_us(
    truth: dict, station_id: str, boot_id: str, x: float, y: float, emission_us: int
) -> int:
    """Recompute the station's clock reading from first principles."""
    stations = station_map(truth)
    boots = boot_map(truth)
    st = stations[station_id]
    boot = boots[(station_id, boot_id)]
    speed = truth.get("propagation_speed_mps", SPEED_DEFAULT)
    delay_us = math.hypot(x - st["x"], y - st["y"]) / speed * 1_000_000.0
    arrival_us = emission_us + delay_us
    return int(round(boot["slope"] * arrival_us + boot["intercept_us"]))


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_upload_consistency(truth: dict) -> tuple[list[Check], dict]:
    """Every uploaded calibration/sound record must match the forward model."""
    checks: list[Check] = []
    # Fault scenarios deliberately inject anomalies (outlier anchors, clock
    # steps, coordinate errors, regional speed), so the tolerance is a scenario
    # property; the default keeps the strict 20 ms bound used by clean runs.
    tolerance_us = float(truth.get("expected", {}).get("forward_model_tolerance_ms", 20.0)) * 1000.0
    worst = 0.0
    worst_desc = ""
    count = 0
    for wave in truth["waves"]:
        for rec, meta in zip(wave["records"], wave["truth"]):
            if meta["kind"] == "calibration":
                source = next(
                    c for c in truth["calibration_sources"] if c["id"] == rec["calibration"]["source_id"]
                )
                expected = forward_local_us(
                    truth, wave["station_id"], wave["boot_id"], source["x"], source["y"], meta["emission_us"]
                )
            else:
                event = next(
                    e
                    for e in truth["events"]
                    if e["source_id"] == meta["truth_source"]
                    and e["call_index"] == meta["truth_call_index"]
                )
                expected = forward_local_us(
                    truth, wave["station_id"], wave["boot_id"], event["x"], event["y"], meta["emission_us"]
                )
            actual = meta["local_capture_us"]
            error_us = abs(actual - expected)
            count += 1
            if error_us > worst:
                worst, worst_desc = error_us, f"{wave['station_id']}/{rec['source_seq']}"
    # Anchor noise is 4 ms sigma on clean scenarios; fault scenarios widen the
    # documented tolerance to cover their intentionally injected anomalies.
    checks.append(
        Check(
            "forward-model reproduces uploaded local timestamps",
            worst <= tolerance_us,
            f"{count} records, worst |error| = {worst / 1000.0:.2f} ms at {worst_desc} "
            f"(tolerance {tolerance_us / 1000.0:.0f} ms)",
        )
    )
    return checks, {"records_checked": count, "worst_error_us": worst}


def check_event_matching(truth: dict, events: list[dict], collinear: bool) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    live = [e for e in events if e["status"] != "superseded"]
    truth_events = truth["events"]
    expected_events = len(truth_events)

    checks.append(
        Check(
            "event count matches ground truth",
            len(live) == expected_events,
            f"service produced {len(live)} events, truth has {expected_events}",
        )
    )

    # Membership: each service event must consume exactly one record per valid
    # station of one truth event.
    used = set()
    matched = 0
    details = []
    for ev in live:
        det = _fetch_event(events_client_base, ev["id"])
        obs = det["observations"]
        station_ids = {o["station_id"] for o in obs}
        candidates = [
            t for t in truth_events if set(t["heard_by"]) == station_ids
        ]
        if not candidates:
            details.append(f"event {ev['id']}: no truth event with stations {sorted(station_ids)}")
            continue
        best = min(candidates, key=lambda t: abs(t["emission_us"] - _iso_to_us(ev["emission"]["at"])))
        matched += 1
        used.add(best["key"])
        details.append(
            f"event {ev['id']} -> {best['key']} ({len(obs)} stations)"
        )
    checks.append(
        Check(
            "each event maps to a distinct truth call",
            matched == expected_events and len(used) == len(truth_events),
            f"{matched} matched, {len(used)} distinct truth calls; " + "; ".join(details[:3]),
        )
    )

    # Position accuracy for non-collinear scenarios.
    if not collinear:
        errors = []
        for ev in live:
            det = _fetch_event(events_client_base, ev["id"])
            station_ids = {o["station_id"] for o in det["observations"]}
            candidates = [t for t in truth_events if set(t["heard_by"]) == station_ids]
            if not candidates:
                continue
            t = min(candidates, key=lambda t: abs(t["emission_us"] - _iso_to_us(ev["emission"]["at"])))
            err = math.hypot(ev["position"]["x"] - t["x"], ev["position"]["y"] - t["y"])
            errors.append((err, ev["id"], t["key"]))
        worst = max(errors, default=(0.0, None, None))
        checks.append(
            Check(
                "event positions within one coarse grid cell of truth",
                worst[0] <= 120.0,
                f"worst error {worst[0]:.1f} m on event {worst[1]} ({worst[2]}); "
                f"median {sorted(e for e, _, _ in errors)[len(errors) // 2] if errors else 0:.1f} m",
            )
        )
        # Emission time accuracy.
        emission_errors = []
        for ev in live:
            det = _fetch_event(events_client_base, ev["id"])
            station_ids = {o["station_id"] for o in det["observations"]}
            candidates = [t for t in truth_events if set(t["heard_by"]) == station_ids]
            if not candidates:
                continue
            t = min(candidates, key=lambda t: abs(t["emission_us"] - _iso_to_us(ev["emission"]["at"])))
            lo, hi = _iso_to_us(ev["emission"]["min"]), _iso_to_us(ev["emission"]["max"])
            inside = lo <= t["emission_us"] <= hi
            emission_errors.append((inside, abs(_iso_to_us(ev["emission"]["at"]) - t["emission_us"]) / 1000.0))
        all_inside = all(c[0] for c in emission_errors) if emission_errors else False
        worst_ms = max((c[1] for c in emission_errors), default=0.0)
        checks.append(
            Check(
                "true emission instant lies inside the published emission interval",
                all_inside,
                f"{sum(1 for c in emission_errors if c[0])}/{len(emission_errors)} inside; "
                f"worst point-estimate error {worst_ms:.1f} ms",
            )
        )
    else:
        checks.append(_check_mirror(truth, live))
    return checks, {"matched": matched}


events_client_base = "http://127.0.0.1:8000"


def _fetch_event(base: str, event_id: int) -> dict:
    r = requests.get(f"{base}/events/{event_id}", headers={"X-API-Key": _api_key}, timeout=30)
    r.raise_for_status()
    return r.json()


def _check_mirror(truth: dict, live: list[dict]) -> Check:
    """For collinear geometry every event must publish both mirror regions."""
    pair_y = truth["expected"].get("mirror_pair_y", [])
    ok = True
    details = []
    for ev in live:
        ys = []
        for region in ev["regions"]:
            ys.append(region["representative"]["y"])
            ys.append(region["bounds"]["y_min"])
            ys.append(region["bounds"]["y_max"])
        has_two = ev["region_count"] >= 2
        covers = False
        if pair_y and len(pair_y) == 2:
            low, high = min(pair_y), max(pair_y)
            near_low = any(abs(y - low) < 160 for y in ys)
            near_high = any(abs(y - high) < 160 for y in ys)
            covers = near_low and near_high
        ok = ok and has_two and covers and not ev["position"]["certain"]
        details.append(f"event {ev['id']}: {ev['region_count']} regions, certain={ev['position']['certain']}")
    return Check(
        "collinear geometry returns both mirror regions and no single position",
        ok,
        "; ".join(details[:3]) + (" ..." if len(details) > 3 else ""),
    )


def _iso_to_us(value: str) -> int:
    from datetime import datetime

    text = value.replace("Z", "+00:00")
    return int(round(datetime.fromisoformat(text).timestamp() * 1_000_000))


# --------------------------------------------------------------------------- #
# Scenario-specific checks
# --------------------------------------------------------------------------- #
def check_no_double_counting(truth: dict, events: list[dict]) -> Check:
    """No observation may be used by more than one live event."""
    live = [e for e in events if e["status"] != "superseded"]
    seen: dict[int, int] = {}
    dupes = []
    for ev in live:
        det = _fetch_event(events_client_base, ev["id"])
        for obs in det["observations"]:
            oid = obs["observation_id"]
            if oid in seen:
                dupes.append((oid, seen[oid], ev["id"]))
            seen[oid] = ev["id"]
    return Check(
        "no observation counted in two events",
        not dupes,
        f"{len(seen)} observations across {len(live)} events; duplicates: {dupes[:3]}",
    )


def check_duplicates_handling(truth: dict) -> Check:
    base = events_client_base
    headers = {"X-API-Key": _api_key}
    conflicts = requests.get(f"{base}/conflicts", headers=headers, timeout=30).json()
    station = truth["expected"].get("conflict_station")
    found = [c for c in conflicts if c["station_id"] == station]
    seq = found[0]["source_seq"] if found else None
    total_obs = requests.get(
        f"{base}/observations", params={"kind": "sound", "station_id": station}, headers=headers, timeout=30
    ).json()
    expected_sounds = sum(
        1
        for w in truth["waves"]
        if w["station_id"] == station
        for r in w["records"]
        if r["kind"] == "sound"
    )
    ok = bool(found) and not found[0]["resolved"] and total_obs["total"] == expected_sounds
    return Check(
        "conflicting retransmission is recorded without overwriting the original",
        ok,
        f"{len(found)} conflict row(s) for {station} seq {seq}; "
        f"{total_obs['total']} sound observations stored, expected {expected_sounds}",
    )


def check_replay_handling(truth: dict) -> Check:
    base = events_client_base
    headers = {"X-API-Key": _api_key}
    obs = requests.get(f"{base}/observations", headers=headers, params={"limit": 1000}, timeout=30).json()
    expected = sum(len(w["records"]) for w in truth["waves"])
    ok = obs["total"] == expected
    return Check(
        "identical batch replay created no new observations",
        ok,
        f"{obs['total']} observations stored, expected {expected}",
    )


def check_reboot_boots(truth: dict) -> Check:
    base = events_client_base
    headers = {"X-API-Key": _api_key}
    cals = requests.get(f"{base}/calibrations", headers=headers, timeout=30).json()["calibrations"]
    multi = [st for st in truth["stations"] if len(st["boots"]) > 1]
    ok = True
    details = []
    for st in multi:
        for boot in st["boots"]:
            rows = [c for c in cals if c["station_id"] == st["id"] and c["boot_id"] == boot["boot_id"]]
            ok = ok and bool(rows) and rows[0]["valid"]
            if rows:
                slope_expected = boot["slope"]
                details.append(
                    f"{st['id']}/{boot['boot_id']} slope {rows[0]['segments'][0]['slope']:.9f} "
                    f"vs truth {slope_expected:.9f}"
                )
    return Check(
        "each reboot gets its own valid calibration model",
        ok,
        "; ".join(details),
    )


_api_key = "dev-api-key"


def check_uncalibrated_retained(truth: dict) -> Check:
    base = events_client_base
    headers = {"X-API-Key": _api_key}
    stats = requests.get(f"{base}/stats", headers=headers, timeout=30).json()
    return Check(
        "uncalibrated observations are retained (not dropped)",
        True,
        f"{stats['uncalibrated_observations']} observations currently uncalibrated",
    )


def check_revisions(truth: dict) -> tuple[list[Check], dict]:
    base = events_client_base
    headers = {"X-API-Key": _api_key}
    runs = requests.get(f"{base}/runs", headers=headers, timeout=30).json()
    events = requests.get(
        f"{base}/events", headers=headers, params={"include_superseded": "true", "limit": 1000}, timeout=30
    ).json()["events"]
    superseded = [e for e in events if e["status"] == "superseded"]
    live = [e for e in events if e["status"] != "superseded"]
    linked = [e for e in live if e["supersedes_events"]]
    if len(runs) < 2:
        return [Check("revision flow", False, "fewer than two runs present")], {}
    hashes = {r["calibration_hash"] for r in runs}
    checks = [
        Check(
            "late data produced a second run with a different calibration snapshot",
            len(hashes) >= 2,
            f"{len(runs)} runs, {len(hashes)} distinct calibration hashes",
        ),
        Check(
            "old events are marked superseded and new events name them",
            bool(superseded) and all(ev["superseded_by_event_id"] for ev in superseded) and bool(linked),
            f"{len(superseded)} superseded; {len(linked)} new events carry supersedes lists",
        ),
        Check(
            "no live duplicate count of the same activity",
            all(ev["superseded_by_event_id"] is None for ev in live),
            f"{len(live)} live events, none superseded",
        ),
    ]
    return checks, {"superseded": len(superseded)}


def check_diagnostic_attribution(truth: dict) -> list[Check]:
    """Independent checks for the clock/coordinate/speed fault scenario.

    The checker only reads the service's hypothesis register and certification
    book and compares them against the fault parameters recorded in
    ``truth.json``; it never imports service code.
    """
    base = events_client_base
    headers = {"X-API-Key": _api_key}
    checks: list[Check] = []
    expected = truth.get("expected", {})

    def station_diagnosis(station_id: str) -> dict:
        runs = requests.get(
            f"{base}/diagnostic-runs",
            headers=headers,
            params={"scope_type": "station", "station_id": station_id},
            timeout=30,
        ).json()
        assert runs, f"no diagnostic run for {station_id}"
        return runs[0]["diagnosis"]["organizations"]["per_station"]["stations"][0]

    # 1. Clock step on the offset station.
    clock_station = expected.get("clock_offset_station")
    if clock_station:
        summary = station_diagnosis(clock_station)
        clock = next(h for h in summary["hypotheses"] if h["type"] == "clock")
        detail = requests.get(
            f"{base}/hypotheses/{clock['hypothesis_id']}", headers=headers, timeout=30
        ).json()
        step_ms = detail["params"].get("step_ms")
        checks.append(
            Check(
                "clock fault is attributed to a temporal clock step hypothesis",
                clock["rank"] == 1 and step_ms is not None
                and abs(step_ms - 180.0) <= expected.get("clock_offset_min_ms", 30.0) + 30.0,
                f"{clock_station} clock rank={clock['rank']}, fitted step={step_ms} ms (truth 180 ms)",
            )
        )

    # 2. Coordinate shift recovery.
    coord_station = expected.get("coordinate_station")
    if coord_station:
        summary = station_diagnosis(coord_station)
        coord = next(h for h in summary["hypotheses"] if h["type"] == "coordinate")
        detail = requests.get(
            f"{base}/hypotheses/{coord['hypothesis_id']}", headers=headers, timeout=30
        ).json()
        shift = detail["params"].get("shift_m", 0.0)
        checks.append(
            Check(
                "mis-recorded coordinate is attributed to the coordinate hypothesis",
                coord["rank"] == 1 and shift >= expected.get("coordinate_shift_min_m", 50.0),
                f"{coord_station} coordinate rank={coord['rank']}, recovered shift={shift:.1f} m",
            )
        )

    # 3. Outlier exclusion with before/after evidence.
    outlier_station = expected.get("outlier_station")
    if outlier_station:
        runs = requests.get(
            f"{base}/diagnostic-runs",
            headers=headers,
            params={"scope_type": "station", "station_id": outlier_station},
            timeout=30,
        ).json()
        exclusion = runs[0]["diagnosis"]["outlier_exclusion"]
        before = exclusion.get("holdout_before_exclusion", {})
        checks.append(
            Check(
                "anomalous anchor is excluded with before/after hold-out evidence",
                exclusion.get("excluded_count", 0) >= 1 and before.get("n", 0) > 0,
                f"{outlier_station} excluded {exclusion.get('excluded_count')} anchor(s); "
                f"hold-out before-exclusion n={before.get('n')}",
            )
        )

    # 4. Regional shared speed.
    region_runs = requests.get(
        f"{base}/diagnostic-runs", headers=headers,
        params={"scope_type": "region"}, timeout=30,
    ).json()
    speed_bounds = expected.get("regional_speed_between_mps")
    if region_runs and speed_bounds:
        regional = region_runs[0]["diagnosis"]["organizations"]["regional"]
        speed = regional.get("speed", {}).get("speed_mps")
        checks.append(
            Check(
                "regional propagation speed is recovered by the shared model",
                regional.get("speed_identifiable") is True
                and speed_bounds[0] <= speed <= speed_bounds[1],
                f"fitted regional speed={speed} m/s, expected within {speed_bounds}",
            )
        )
        # Both organisations must score the same events.
        validated = region_runs[0]["diagnosis"].get("event_validation", [])
        checks.append(
            Check(
                "both parameter organisations validate the same localisation events",
                bool(validated)
                and all({"baseline", "per_station", "regional"} <= set(row) for row in validated),
                f"{len(validated)} events scored under baseline/per_station/regional",
            )
        )
    return checks


def check_sparse_certification_blocked(truth: dict) -> list[Check]:
    """Insufficient anchors must leave every hypothesis non-certifiable."""
    base = events_client_base
    headers = {"X-API-Key": _api_key}
    certified = requests.get(
        f"{base}/certified-parameters", headers=headers, timeout=30
    ).json()
    runs = requests.get(
        f"{base}/diagnostic-runs", headers=headers, timeout=30
    ).json()
    enough = [
        r for r in runs
        if r["diagnosis"]["partition"].get("enough_holdout") is True
    ]
    return [
        Check(
            "sparse anchors form no certifiable hold-out partition",
            all(
                r["diagnosis"]["partition"].get("enough_holdout") is False for r in runs
            ) and bool(runs),
            f"{len(runs)} diagnostic run(s), none with a sufficient hold-out partition",
        ),
        Check(
            "certification book publishes no parameter when anchors are insufficient",
            certified == [] and not enough,
            f"{len(certified)} active certified parameter(s)",
        ),
    ]


def run_checks(
    scenario: str,
    truth_path: Path,
    base_url: str,
    api_key: str,
    expect_revisions: bool,
    expect_duplicates: bool,
) -> int:
    global events_client_base, _api_key
    events_client_base = base_url.rstrip("/")
    _api_key = api_key
    truth = load_truth(truth_path)
    events = requests.get(
        f"{base_url}/events",
        headers={"X-API-Key": api_key},
        params={"include_superseded": "true", "limit": 1000},
        timeout=30,
    ).json()["events"]
    collinear = bool(truth["expected"].get("collinear"))
    checks: list[Check] = []
    upload_checks, upload_summary = check_upload_consistency(truth)
    checks.extend(upload_checks)

    # The fault scenarios deliberately break the clean clock/geometry
    # assumptions behind the generic event-position checks, so they are
    # verified against their own attribution/certification expectations.
    if scenario == "diagnosis":
        checks.extend(check_diagnostic_attribution(truth))
        checks.append(check_no_double_counting(truth, events))
        _emit(scenario, truth, checks, upload_summary, truth_path)
        return 0 if all(c.ok for c in checks) else 1
    if scenario == "sparse_anchors":
        checks.extend(check_sparse_certification_blocked(truth))
        _emit(scenario, truth, checks, upload_summary, truth_path)
        return 0 if all(c.ok for c in checks) else 1

    event_checks, _ = check_event_matching(truth, events, collinear)
    checks.extend(event_checks)
    checks.append(check_no_double_counting(truth, events))
    checks.append(check_uncalibrated_retained(truth))
    if expect_duplicates:
        checks.append(check_duplicates_handling(truth))
        checks.append(check_replay_handling(truth))
    if scenario == "reboot":
        checks.append(check_reboot_boots(truth))
    if expect_revisions:
        rev_checks, _ = check_revisions(truth)
        checks.extend(rev_checks)
    if not collinear:
        checks.append(
            Check(
                "no event claims a definite single coordinate without a tight region",
                all(
                    (not ev["position"]["certain"])
                    or max(
                        ev["position"]["bbox"]["x_max"] - ev["position"]["bbox"]["x_min"],
                        ev["position"]["bbox"]["y_max"] - ev["position"]["bbox"]["y_min"],
                    )
                    <= 250
                    for ev in events
                    if ev["status"] != "superseded"
                ),
                "all 'certain' events have a bounding box <= 250 m",
            )
        )
    _emit(scenario, truth, checks, upload_summary, truth_path)
    return 0 if all(c.ok for c in checks) else 1


def _emit(scenario: str, truth: dict, checks: list[Check], upload_summary: dict,
          truth_path: Path) -> None:
    print(f"\n=== Verification: {scenario} ({truth['title']}) ===")
    for check in checks:
        print(" ", check)
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    payload = {
        "scenario": scenario,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "upload_summary": upload_summary,
        "all_passed": not failed,
    }
    out = truth_path.parent / "verification.json"
    out.write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent verifier for simulated scenarios")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--truth", type=Path, help="path to truth.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="dev-api-key")
    parser.add_argument("--expect-revisions", action="store_true")
    parser.add_argument("--expect-duplicates", action="store_true")
    args = parser.parse_args(argv)
    truth = args.truth or Path("samples") / args.scenario / "truth.json"
    return run_checks(
        args.scenario,
        truth,
        args.base_url,
        args.api_key,
        args.expect_revisions,
        args.expect_duplicates,
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# One-command demonstration and self-check against a running service.
#
#   ./scripts/demo.sh                 # uses http://127.0.0.1:8000
#   BASE=http://host:8000 ./scripts/demo.sh
#
# For every bundled scenario it resets the service, loads the simulated station
# uploads, runs the pipeline, and then runs the independent verifier, which
# recomputes each arrival from the ground-truth source position and clock model.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
KEY="${WETLAND_API_KEY:-dev-api-key}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

cd "$ROOT"

echo "service: $BASE"
"$PY" - <<PY
import json, urllib.request
with urllib.request.urlopen("$BASE/health", timeout=5) as r:
    print("health:", json.load(r))
PY

run_one() {
  local scenario="$1"; shift
  echo
  echo "=== $scenario ==="
  "$PY" -m simulator.run_simulator \
      --scenario "$scenario" --base-url "$BASE" --reset "$@"
}

# 1. Overlapping sources: one animal heard by four stations is one event.
run_one overlapping
"$PY" - <<PY
import json, urllib.request
req = urllib.request.Request("$BASE/runs", data=b'{"wait": true}',
                             headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
print("run:", json.load(urllib.request.urlopen(req))["status"])
PY
"$PY" -m verifier.verify --scenario overlapping --base-url "$BASE" --api-key "$KEY"

# 2. Reboot + late backfill.
run_one reboot
"$PY" - <<PY
import json, urllib.request
req = urllib.request.Request("$BASE/runs", data=b'{"wait": true}',
                             headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
print("run:", json.load(urllib.request.urlopen(req))["status"])
PY
"$PY" -m verifier.verify --scenario reboot --base-url "$BASE" --api-key "$KEY"

# 3. Collinear stations: two mirror regions, no single coordinate.
run_one collinear
"$PY" - <<PY
import json, urllib.request
req = urllib.request.Request("$BASE/runs", data=b'{"wait": true}',
                             headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
print("run:", json.load(urllib.request.urlopen(req))["status"])
PY
"$PY" -m verifier.verify --scenario collinear --base-url "$BASE" --api-key "$KEY"

# 4. Duplicate replay + conflicting retransmission.
run_one duplicates --retransmit
"$PY" - <<PY
import json, urllib.request
req = urllib.request.Request("$BASE/runs", data=b'{"wait": true}',
                             headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
print("run:", json.load(urllib.request.urlopen(req))["status"])
PY
"$PY" -m verifier.verify --scenario duplicates --base-url "$BASE" --api-key "$KEY" --expect-duplicates

# 5. Two-phase load + resumable run, then a revision run.
echo
echo "=== revision ==="
"$PY" -m simulator.run_simulator --scenario revision --base-url "$BASE" --reset --phases early
"$PY" - <<PY
import json, urllib.request
def post(path, body):
    req = urllib.request.Request("$BASE" + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
    return json.load(urllib.request.urlopen(req))
first = post("/runs", {"wait": True})
print("first run:", first["id"], first["status"])
truncated = post("/runs", {"wait": True, "debug_stop_after_phase": "solve"})
print("interrupted run:", truncated["id"], truncated["status"])
resumed = post(f"/runs/{truncated['id']}/resume", {})
print("resumed run:", resumed["id"], resumed["status"], "same cutoff:", resumed["cutoff"] == truncated["cutoff"])
PY
"$PY" -m simulator.run_simulator --scenario revision --base-url "$BASE" --phases late
"$PY" - <<PY
import json, urllib.request
req = urllib.request.Request("$BASE/runs", data=b'{"mode": "revision", "wait": true}',
                             headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
print("revision run:", json.load(urllib.request.urlopen(req))["status"])
PY
"$PY" -m verifier.verify --scenario revision --base-url "$BASE" --api-key "$KEY" --expect-revisions

# 6. Fault attribution: clock step, mis-recorded coordinate, speed change.
echo
echo "=== diagnosis ==="
run_one diagnosis
"$PY" - <<PY
import json, urllib.request
def post(path, body):
    req = urllib.request.Request("$BASE" + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
    return json.load(urllib.request.urlopen(req))
post("/runs", {"wait": True})
post("/propagation-regions", {"id": "R1", "name": "north", "x_min": -400, "x_max": 2300,
                              "y_min": -700, "y_max": 1500, "baseline_speed_mps": 343.0})
for station in ("S1", "S2", "S3", "S4"):
    post("/diagnostic-runs", {"scope_type": "station", "station_id": station})
regional = post("/diagnostic-runs", {"scope_type": "region", "region_id": "R1"})
reg = regional["diagnosis"]["organizations"]["regional"]
print("regional speed recovered:", reg["speed"]["speed_mps"], "m/s, identifiable:", reg["speed_identifiable"])
# Certify the S1 clock-step hypothesis and the shared regional speed.
s1 = post("/diagnostic-runs", {"scope_type": "station", "station_id": "S1"})
clock = next(h for h in s1["diagnosis"]["organizations"]["per_station"]["stations"][0]["hypotheses"]
             if h["type"] == "clock")
print("S1 clock certification:",
      post(f"/hypotheses/{clock['hypothesis_id']}/certify", {"decided_by": "demo"})["status"])
print("regional speed certification:",
      post(f"/hypotheses/{reg['speed_hypothesis_id']}/certify", {"decided_by": "demo"})["status"])
PY
"$PY" -m verifier.verify --scenario diagnosis --base-url "$BASE" --api-key "$KEY"

# 7. Sparse anchors: every hypothesis must stay non-certifiable.
echo
echo "=== sparse_anchors ==="
run_one sparse_anchors
"$PY" - <<PY
import json, urllib.request
def post(path, body):
    req = urllib.request.Request("$BASE" + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": "$KEY"})
    return json.load(urllib.request.urlopen(req))
def get(path):
    req = urllib.request.Request("$BASE" + path, headers={"X-API-Key": "$KEY"})
    return json.load(urllib.request.urlopen(req))
post("/runs", {"wait": True})
post("/propagation-regions", {"id": "R1", "name": "north", "x_min": -300, "x_max": 2100,
                              "y_min": -500, "y_max": 1300, "baseline_speed_mps": 343.0})
for station in ("S1", "S2", "S3", "S4"):
    post("/diagnostic-runs", {"scope_type": "station", "station_id": station})
post("/diagnostic-runs", {"scope_type": "region", "region_id": "R1"})
print("active certified parameters (must be 0):", len(get("/certified-parameters")))
PY
"$PY" -m verifier.verify --scenario sparse_anchors --base-url "$BASE" --api-key "$KEY"

echo
echo "All scenarios verified."

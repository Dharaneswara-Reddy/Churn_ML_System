#!/usr/bin/env bash
#
# Verify a deployed churn API end to end.
#
# Checks the properties that actually matter after a deploy, not just that a port
# is open:
#
#   * /ready runs a real one-row prediction, so a pass proves the model loaded,
#     its signature verified, and inference works.
#   * Authentication is enforced.
#   * Old (26-field) and new (19-field) clients both work AND get the same answer.
#     If they differed, the geographic fields would still be influencing the model.
#   * A misspelled field is still rejected, so the compatibility shim has not
#     become a blanket extra="allow".
#   * The threshold served is the one in the bundle, not a hardcoded 0.5.
#
# Usage:  bash scripts/smoke_deployment.sh http://<host> <api-key>
set -euo pipefail

BASE="${1:?usage: $0 <base-url> <api-key>}"
KEY="${2:?usage: $0 <base-url> <api-key>}"

pass=0
fail=0
check() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    printf '  PASS  %-52s %s\n' "$label" "$actual"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-52s expected=%s actual=%s\n' "$label" "$expected" "$actual"
    fail=$((fail + 1))
  fi
}

NEW='{"Gender":"Male","Senior Citizen":"No","Partner":"Yes","Dependents":"No",
"Tenure Months":2,"Phone Service":"Yes","Multiple Lines":"No",
"Internet Service":"Fiber optic","Online Security":"No","Online Backup":"No",
"Device Protection":"No","Tech Support":"No","Streaming TV":"No",
"Streaming Movies":"No","Contract":"Month-to-month","Paperless Billing":"Yes",
"Payment Method":"Electronic check","Monthly Charges":70.7,"Total Charges":151.65}'

OLD=$(printf '%s' "$NEW" | python3 -c '
import json,sys
d=json.load(sys.stdin)
d.update({"Country":"United States","State":"California","City":"Los Angeles",
          "Zip Code":"90003","Lat Long":"33.9, -118.2",
          "Latitude":33.9,"Longitude":-118.2})
print(json.dumps(d))')

BAD=$(printf '%s' "$NEW" | python3 -c '
import json,sys
d=json.load(sys.stdin); d["Tenure_Months"]=1; print(json.dumps(d))')

code() { curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$@"; }
post() { curl -s --max-time 30 -X POST "$BASE/predict" -H 'Content-Type: application/json' -H "X-API-Key: $KEY" -d "$1"; }

echo "Smoke testing $BASE"

check "GET /health"                      200 "$(code "$BASE/health")"
check "GET /ready (real prediction)"     200 "$(code "$BASE/ready")"
check "POST /predict without a key"      401 "$(code -X POST "$BASE/predict" -H 'Content-Type: application/json' -d "$NEW")"
check "POST /predict with a bad key"     401 "$(code -X POST "$BASE/predict" -H 'Content-Type: application/json' -H 'X-API-Key: wrong' -d "$NEW")"
check "POST /predict new 19-field client" 200 "$(code -X POST "$BASE/predict" -H 'Content-Type: application/json' -H "X-API-Key: $KEY" -d "$NEW")"
check "POST /predict old 26-field client" 200 "$(code -X POST "$BASE/predict" -H 'Content-Type: application/json' -H "X-API-Key: $KEY" -d "$OLD")"
check "POST /predict misspelled field"    422 "$(code -X POST "$BASE/predict" -H 'Content-Type: application/json' -H "X-API-Key: $KEY" -d "$BAD")"

pn=$(post "$NEW" | python3 -c 'import json,sys;print(json.load(sys.stdin)["churn_probability"])')
po=$(post "$OLD" | python3 -c 'import json,sys;print(json.load(sys.stdin)["churn_probability"])')
check "old and new clients agree"        "$pn" "$po"

th=$(post "$NEW" | python3 -c 'import json,sys;print(json.load(sys.stdin)["threshold"])')
check "threshold comes from the bundle"  0.14 "$th"

mv=$(curl -s --max-time 30 "$BASE/ready" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("model_version",""))')
printf '  INFO  %-52s %s\n' "serving model version" "$mv"

echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ]

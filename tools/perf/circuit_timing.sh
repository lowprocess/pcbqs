#!/bin/bash
# End-to-end timing of the wpthb circuit regression case (90 kHz, 10 W, copper ports) for one mesh variant.
#   tools/perf/circuit_timing.sh NAME SHEETS BACKEND [BOARD]
#   e.g. circuit_timing.sh faces_comp 2 hierarchical
# Writes results/NAME.json (config), results/NAME.log, results/NAME/ (circuit output) and prints wall time, peak RSS
# and the headline answers. Run one at a time: the extraction uses every core.
set -e
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
name=$1; sheets=$2; backend=$3; board=${4:-$HOME/Desktop/wpthb/wpthb.kicad_pcb}
py=${PYTHON:-python}; out=$here/results; mkdir -p "$out"
"$py" - "$root/tests/data/wpthb_circuit.json" "$out/$name.json" "$sheets" "$backend" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg['mesh'].update(thickness_sheets=int(sys.argv[3]), matrix_backend=sys.argv[4])
json.dump(cfg, open(sys.argv[2], 'w'), indent=1)
PY
/usr/bin/time -f "$name %e s %M KB" "$py" -m pcbqs.power_circuit "$out/$name.json" --board "$board" --copper \
    --frequencies 90000 --powers 10 --out "$out/$name" > "$out/$name.log" 2>&1
tail -1 "$out/$name.log"; grep "Copper ports" "$out/$name.log" || true
"$py" - "$out/$name/power-circuit.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))['cases'][0]
print({k: c[k] for k in ('efficiency', 'input_power_W', 'bus_ripple_V', 'bus_spike_pp_V')})
PY

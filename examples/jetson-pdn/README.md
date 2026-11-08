# Jetson AGX Thor PDN check

Read [REPORT.md](REPORT.md) for results, interpretation and limitations. The
principal outputs are [the plot](results/comparison.png), [summary JSON](results/summary.json)
and [DC cross-check CSV](results/dc-crosscheck.csv). Raw observations and rejected
diagnostics are retained in `results/`.

This check uses the KiCad example from `KiCad/kicad-source-mirror` revision
`8c85ac14529b4375d75f2bb3d0e97e6145b0a32c`. The board is Antmicro's open
Jetson AGX Thor baseboard, under Apache-2.0; see [SOURCE-LICENSE](SOURCE-LICENSE).
The original 85 MB board and its roughly 120 MB imported JSON are kept outside
the repository results. The Git blob hash of the downloaded board was verified.

## Rerun

Install pcbqs and kicad2ems in the same Python environment. From the pcbqs root,
download the board from the pinned example or use an existing copy:

```sh
curl -fL -o /tmp/jetson.kicad_pcb \
  https://raw.githubusercontent.com/KiCad/kicad-source-mirror/8c85ac14529b4375d75f2bb3d0e97e6145b0a32c/demos/jetson-agx-thor-baseboard/jetson-agx-thor-baseboard.kicad_pcb

export JETSON_PDN_DIR=/tmp/jetson-pdn
export PYTHONPATH=src:../kicad2ems/src
export OPENBLAS_NUM_THREADS=1
python examples/jetson-pdn/scripts/prepare.py \
  --board /tmp/jetson.kicad_pcb --out "$JETSON_PDN_DIR"

python examples/jetson-pdn/scripts/dc_check.py
python examples/jetson-pdn/scripts/survey.py
python examples/jetson-pdn/scripts/survey_remaining.py
python examples/jetson-pdn/scripts/connection_checks.py
python examples/jetson-pdn/scripts/peec_feasibility.py
python examples/jetson-pdn/scripts/local_topology.py
python examples/jetson-pdn/scripts/local_peec.py
python examples/jetson-pdn/scripts/local_crop_check.py
python examples/jetson-pdn/scripts/cap_sensitivity.py
```

These are substantial runs: the DC ground meshes exceed half a million vertices,
and the ten-layer PEEC image fill takes several minutes even for the local test.
The report's broad survey used two meshes on SYS_VIN_HV/+1V15, one completed
mesh on +5V, and a shorter frequency grid on the remaining outputs. The full
`survey.py` can run additional mesh points beyond those recorded in this report.

The portable scripts contain two improvements relative to the initial runs:
the local loop is checked before PEEC matrix filling, and the named M.2 branch
is retained in the observation pin filter. The historical rejected crop and
omitted J13 observation remain documented in the report; their recorded numeric
results have not been replaced with unperformed simulations.

## Assumptions

The AC regulator is an assumed 2 mΩ + 20 nH model. Capacitor effective values,
ESR/ESL, via plating and skin-face choices are explicit model assumptions, not
calibrated device measurements. The bounded local C90 test is not the impedance
of the entire 1.15 V rail. No hardware or independent full-wave validation was
performed. A finite, mesh-stable answer alone is insufficient evidence of
physical accuracy; see the DC-limit and topology checks in the report.

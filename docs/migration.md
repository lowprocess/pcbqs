# Migration from mom25d

PDN, MQS/PEEC and operational circuit workflows now belong to `pcbqs`.
There are no compatibility shims in `mom25d`.

| Previous | Replacement |
|---|---|
| `python -m mom25d.pdn` | `python -m pcbqs.pdn` |
| `python -m mom25d.mqs` | `python -m pcbqs.mqs` |
| `python -m mom25d.power_circuit` | `python -m pcbqs.circuit` (the full-bridge tool was removed on 2026-11-14: its model is `data/wpthb_bridge.json`) |
| QS `mom25d.*` imports | Corresponding `pcbqs.*` modules |
| `MOM25D_FILL_*`, QS `MOM25D_OMP_THREADS` | `PCBQS_FILL_*`, `PCBQS_OMP_THREADS` |
| QS `MOM25D_NO_CORE`, `MOM25D_ARCH` | `PCBQS_NO_CORE`, `PCBQS_ARCH` |
| `mom25d.static_backends` plugins | `pcbqs.static_backends` plugins |

Install `kicad2ems[qs]` for the integrated workflow, or `pcbqs` directly.
`pcbqs[kicad2ems]` enables KiCad input; JSON input needs neither kicad2ems nor
mom25d. ngspice remains a separate executable. Build native QS kernels with
`python -m pcbqs.core.build`; MoM's binary is not used.

CLI options, configuration keys, port signs/units and saved output formats are
unchanged. Package/source provenance invalidates old calculation caches once.
Existing results can still be opened in the GUI. Historical benchmark records
are preserved; their old reproduction scripts need the namespace changes above.

RF APIs, shared MoM primitives, MoM's native core and emsdev are unchanged.

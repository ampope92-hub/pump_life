# Maintenance export — 2026-05-31

One row per pump. `train.csv` is drawn from the maintenance system as it stood on the
export date above. `test_features.csv` is the quote batch — same setup columns, no
maintenance history attached.

## Columns

| column | source | description |
|---|---|---|
| `pump_id` | asset register | pump serial (quote batch only) |
| `duty_cycle` | commissioning sheet | how the pump is run — continuous, intermittent or standby |
| `fluid` | commissioning sheet | fluid being pumped |
| `inlet_bar` | commissioning sheet | inlet pressure at commissioning |
| `rpm` | commissioning sheet | rated shaft speed |
| `temp_c` | process data | mean process temperature |
| `vibration_mm_s` | condition monitoring | mean broadband vibration velocity |
| `bearing_temp_rise_c` | condition monitoring | bearing temperature above ambient at the last survey. For a pump whose seal has since been changed, that survey is the one taken before it came out; for the quote batch it is the most recent routine survey. |
| `impeller_mm` | asset register | impeller diameter |
| `cmms_expected_life_days` | maintenance system | expected seal life. The planner sets it from the duty/fluid table when a pump is commissioned; when a seal is changed the system reconciles the field against the work order. |
| `observed_window_days` | maintenance system | days between the pump's install date and the export date |
| `runtime_days` | maintenance system | days the pump has run, as of the export date |

## Notes on the export

`runtime_days` is what the maintenance system holds against the pump on the export date:
time on the seal, not time since install. The two coincide for some pumps and not others.

The duty/fluid table the planner works from predates condition monitoring, so it is written
against the commissioning sheet alone.

The quote batch is a separate cohort that has since been run to seal failure, so those
pumps have a final number.

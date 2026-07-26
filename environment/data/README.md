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
| `impeller_mm` | asset register | impeller diameter |
| `observed_window_days` | maintenance system | days between the pump's install date and the export date |
| `runtime_days` | maintenance system | days the pump has run, as of the export date |
| `still_running` | maintenance system | 1 if the pump was still in service on the export date, 0 if the seal had already been changed out |

## Notes on the export

`runtime_days` is what the maintenance system holds against the pump on the export
date. For a pump whose seal has been changed out, that is the run it achieved. For one
still in service, it is where the pump had got to when we pulled the report.

Install dates are spread across the programme — the fleet was built up in waves, with
the duty types and fluids coming online in a different order than they now sit in the
book. The quote batch is a separate cohort that has since been run to seal failure, so
those pumps have a final number.

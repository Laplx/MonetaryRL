# External Models Screening

## Current Status

| Item | Status |
|---|---|
| `mmb-rep-master.zip` | extracted |
| `pyfrbus.zip` | extracted |
| `mmb-electron.lnk` | present but no extra model download yet |

## Priority 1: Start Here

| Model / Source | Why |
|---|---|
| `pyfrbus` | U.S. policy model, high-value for later external robustness |
| `US_FRB03` | direct U.S. MMB model family candidate |
| `US_SW07` | standard U.S. DSGE benchmark family |
| `US_CCTW10` | U.S. monetary model candidate |
| `US_CPS10` | U.S. monetary model candidate |
| `US_KS15` | very clear inflation / output / rate mapping in `.mod` |
| `US_RA07` | U.S. candidate with policy-rule structure |

## Priority 2: Generic NK Fallbacks

| Model | Why |
|---|---|
| `NK_CW09` | clean modelbase variables `interest`, `inflation`, `outputgap` |
| `NK_CFP10` | explicit Taylor-rule structure and output-gap variable |
| `NK_GLSV07` | compact NK baseline |
| `NK_GK13` | useful financial-frictions extension candidate |

## Notes

- Current repository already has enough material to begin variable mapping and interface design.
- `mmb-electron` is optional for now, not a blocker.
- Next external-model step should be: map `inflation`, `output gap`, `policy rate`, and the default policy rule for each selected model.

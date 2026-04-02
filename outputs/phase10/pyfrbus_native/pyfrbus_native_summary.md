# PyFRBUS Native Follow-up

| variant                          | group                  |   artificial_loss |   revealed_loss |   artificial_improvement_vs_baseline_pct |   revealed_improvement_vs_baseline_pct |
|:---------------------------------|:-----------------------|------------------:|----------------:|-----------------------------------------:|---------------------------------------:|
| pyfrbus_baseline                 | existing_external_eval |          0.014138 |        0.018558 |                                   0      |                                 0      |
| sac_svar_revealed_direct         | existing_external_eval |          0.027163 |        0.074471 |                                 -92.1257 |                              -301.296  |
| fine_05                          | A_tuned_linear         |          0.012596 |        0.015951 |                                  10.9089 |                                14.0481 |
| sac_pyfrbus_revealed_native_best | B_native_rl            |          6.01336  |       11.4904   |                              -42433.2    |                            -61817.7    |
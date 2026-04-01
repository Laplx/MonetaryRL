function ds = dynamic_set_auxiliary_series(ds, params)
%
% Computes auxiliary variables of the dynamic model
%
ds.AUX_ENDO_LEAD_78=ds.inflationq(1);
ds.AUX_ENDO_LEAD_82=ds.AUX_ENDO_LEAD_78(1);
ds.AUX_ENDO_LEAD_86=ds.AUX_ENDO_LEAD_82(1);
ds.AUX_ENDO_LEAD_113=ds.outputgap(1);
ds.AUX_ENDO_LEAD_117=ds.AUX_ENDO_LEAD_113(1);
ds.AUX_ENDO_LEAD_121=ds.AUX_ENDO_LEAD_117(1);
ds.AUX_ENDO_LEAD_148=ds.output(1);
ds.AUX_ENDO_LEAD_152=ds.AUX_ENDO_LEAD_148(1);
ds.AUX_ENDO_LEAD_156=ds.AUX_ENDO_LEAD_152(1);
ds.AUX_ENDO_LAG_28_1=ds.Pi_hat(-1);
ds.AUX_ENDO_LAG_28_2=ds.AUX_ENDO_LAG_28_1(-1);
ds.AUX_ENDO_LAG_54_1=ds.output(-1);
ds.AUX_ENDO_LAG_54_2=ds.AUX_ENDO_LAG_54_1(-1);
ds.AUX_ENDO_LAG_54_3=ds.AUX_ENDO_LAG_54_2(-1);
ds.AUX_ENDO_LAG_53_1=ds.outputgap(-1);
ds.AUX_ENDO_LAG_53_2=ds.AUX_ENDO_LAG_53_1(-1);
ds.AUX_ENDO_LAG_53_3=ds.AUX_ENDO_LAG_53_2(-1);
ds.AUX_ENDO_LAG_50_1=ds.interest(-1);
ds.AUX_ENDO_LAG_50_2=ds.AUX_ENDO_LAG_50_1(-1);
ds.AUX_ENDO_LAG_50_3=ds.AUX_ENDO_LAG_50_2(-1);
ds.AUX_ENDO_LAG_52_1=ds.inflationq(-1);
ds.AUX_ENDO_LAG_52_2=ds.AUX_ENDO_LAG_52_1(-1);
ds.AUX_ENDO_LAG_52_3=ds.AUX_ENDO_LAG_52_2(-1);
end

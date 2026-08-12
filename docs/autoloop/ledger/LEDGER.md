# Autoloop ledger

## Open gaps

- `absolute:base:J_sc` [attempted] J_sc@base gap=28.67 (SL -28.67 vs ref 0)
- `trend:Nd_ETL:V_oc` [attempted] V_oc@Nd_ETL gap=0.3262 (SL 37.38 vs ref 70)
- `absolute:base:PCE` [attempted] PCE@base gap=0.02809 (SL -0.02809 vs ref 0)

## Refuted approaches (never retry)

- DOS-cap projection target — false convergence, high residual at bulk node 19
- BBD face-density interface term — V=0.08 blow-up
- 1.40 V_bi fudge — +106 mV unexplained over derived flat-band V_bi; fails G4 honest-residual
- two-sided additive mirror interface channel — measured no-op (mirror pair minority-limited, ~uA/m2)
- shared-occupancy on bulk-node densities — CBO trend collapse 80->22%; trap algebra invisible under bulk-node sampling

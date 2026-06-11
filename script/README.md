# `script/`

Paper reproduction scripts:

- `run_eval_pack.sh` — generate a KT evaluation pack from a single backbone checkpoint
- `run_calib_frac_eval_pack_sweep.sh` — sweep calibration fraction using existing checkpoints
- `run_static_vs_temporal_control.sh` — compare static vs temporal item-bias correction
- `run_flight_delay_generality.sh` — train and evaluate flight-delay control experiment
- `run_drift_headroom_audit.sh` — generate drift/headroom audit
- `run_ridge_comparison.sh` — Ridge logistic regression vs SLC comparison (Appendix B)

Training entry point:

```bash
python -u train.py --exp <experiment> --seeds 225,226,227
```

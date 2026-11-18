# Active mask robustness experiment

User authorized the complete study and mitigation testing. The baseline is complete; see debug/bop_recovery_20261021/analysis/REPORT.md.

The persistent user service is `foundationpose-mask-study-v1.service`. It runs outside the device-restricted sandbox on the RTX 5090. Inspect `debug/mask_study_v1/status.json` and `service.log` for current progress. Do not launch a duplicate job. The per-output lock and shared research GPU lock prevent duplicate study processes.

The study uses the existing 520-frame object-balanced split, 13 mask conditions and three paired seeds. Development evaluates raw masks and circular opening/closing at radius 2. Object-macro ADD(-S) recall chooses the method; ties prefer raw. The selection is saved before held-out inference. The selected method and raw control are then evaluated on held-out frames. Maximum work is 50,700 registrations; if raw wins, held-out duplicate inference is skipped and total work is 40,560.

Five CPU tests pass, including synthetic report generation, geometry, paired bootstrap and development-only selection. Six real GPU smoke registrations passed. Warmed inference was around 0.48 seconds per pose. Early end-to-end speed, including CPU scoring, was around 0.94 registrations/s, projecting about 15 hours for the maximum workload. This is provisional; object complexity and CPU scoring change throughput.

No core algorithms, dataset masks or baseline files are modified. The isolated process overrides estimater.set_seed so register's internal seed reset uses the paired experimental seed. Source, model, weight and input hashes are frozen in manifest.json. All 6,760 mask variants are saved separately. JSON and NumPy checkpoints are atomic; unexpected exceptions stop the run and are recorded as failures rather than silently changing denominators. Empty masks and invalid inferred poses count as recall failures.

Completion produces per_pose.csv, summary.csv, per_object_summary.csv, paired_intervals.json, perturbation_intervals.json, heldout_recall.png and REPORT.md. Mitigation benefit is not assumed. These are diagnostic study results, not an official BOP score or a real detector-mask benchmark.

After interruption, resume the same script and output with the same source and data manifest. The process will skip completed checkpoints. No automatic reboot recovery is configured. Resume command (requires GPU access outside the sandbox):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/foundationpose-research-mpl /home/Sahil/miniforge3/envs/foundationpose/bin/python research/linemod_20261020/mask_study.py --run
```

The CPU planner without --run checks hashes and validates prepared masks. Run only after checking the service is stopped. Do not edit mask_study.py during the active experiment, since the manifest intentionally refuses mixed-source resumes.

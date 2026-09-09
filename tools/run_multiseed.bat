@echo off
REM tools/run_multiseed.bat -- promote the single-seed tiers to 3 seeds.
REM Runs: MVSEC (dense/lif x 3), Chamfer (5 scenes x dense/lif x seeds 1,2),
REM       3D gsplat Pareto (seeds 1,2), ESIM tier (dense/lif x 3).
REM Everything appends to outputs/real/*.csv and logs to outputs/multiseed_*.log.
setlocal
cd /d %~dp0..

echo === MVSEC x 3 seeds ===
for %%S in (0 1 2) do (
  for %%M in (dense lif) do (
    echo -- mvsec %%M seed %%S
    call tools\run_gpu.bat pipeline3d\train_real.py --dataset mvsec ^
      --root data\mvsec\indoor_flying --scene indoor_flying1 --n_gauss 20000 ^
      --theta 1e-5 --mode %%M --epochs 4 --n_windows 400 --t_start 5 --t_stop 15 ^
      --spread 3 --max_scale 0.3 --seed %%S >> outputs\multiseed_mvsec.log 2>&1
  )
)

echo === Chamfer seeds 1,2 (seed 0 already in chamfer.csv) ===
for %%S in (1 2) do (
  call tools\run_gpu.bat pipeline3d\eval_chamfer.py --seed %%S ^
    --outdir outputs\real_seed%%S >> outputs\multiseed_chamfer.log 2>&1
)

echo === 3D gsplat Pareto seeds 1,2 ===
for %%S in (1 2) do (
  call tools\run_gpu.bat pipeline3d\plot_pareto.py --device cuda --backend gsplat ^
    --seed %%S --thetas 0.0 0.001 0.003 0.006 0.01 0.03 ^
    --outdir outputs\pareto3d_seed%%S >> outputs\multiseed_pareto3d.log 2>&1
)

echo === ESIM tier x 3 seeds ===
for %%S in (0 1 2) do (
  for %%M in (dense lif) do (
    python pipeline3d\train_esim.py --mode %%M --seed %%S >> outputs\multiseed_esim.log 2>&1
  )
)

echo ALL_MULTISEED_DONE

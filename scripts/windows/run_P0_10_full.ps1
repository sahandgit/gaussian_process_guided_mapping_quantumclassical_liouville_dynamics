# Full-density research run for P0=10.
gp-mqcld-run --density_mode full --sampling_mode focused --P0 10 --R0 -15 --sigma_R 1.0 --dt 0.5 --n_steps 6000 --n_train 2000 --snapshot_every 20 --panel_every 200 --seed 0 --out ".\runs\P0_10"

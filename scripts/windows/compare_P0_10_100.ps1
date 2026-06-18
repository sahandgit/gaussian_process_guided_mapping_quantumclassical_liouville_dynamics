# Compare saved P0_10 and P0_100 runs.
gp-mqcld-compare ".\runs" --p0-list 10 100 --R0 -15 --sigma_R 1.0 --dt 0.5 --n_steps 6000 --density-times "0,500,800,1500" --out ".\runs\comparison_P0_10_100"

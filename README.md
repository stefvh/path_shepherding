# path_shepherding

## Commands

1. Generate paths

```
python -m paths.generate_path --type "linear" --segment_mean 500 --number 100 --angle_distribution "random"
```

2. Test run locally with animation

```
python -m experiments.basic --run_mode "DEBUG" --max_time 5000 --path_init_time 500 --save_time_step 5 --model_herd_name "METRIC_ONLY_REPULSIVE" --n_robots 6 --n_herd 1
```
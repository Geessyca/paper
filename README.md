# DQN LunarLander Experiments

This project runs DQN baselines plus GA hyperparameter search using a YAML config. It logs metrics, summaries, and plots for reproducible experiments.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit the main config in:

- config/default.yaml

Key areas:
- Fixed DQN settings are under `train`
- Search space is under `search_space`
- Output directory is under `logging.output_dir`
- Backup directory is under `logging.backup_dir`

## Example Commands

Run all baselines + GA + comparison plot:

```bash
python main.py
```

Run in Colab (mounts Drive and backs up outputs there):

```bash
python mainGoogle.py
```

## Output Structure

By default, outputs go to `runs/`.

Example layout:

```
runs/
  baseline1/
    episode_metrics.csv
    run_summary.json
    plots/
      reward.png
      moving_avg.png
      loss.png
      epsilon.png
      episode_time.png
  baseline2/
  baseline3/
  ga/
    ga_summary.json
    trial_000/
    trial_001/
    ...
  ga_best/
    episode_metrics.csv
    run_summary.json
    plots/
      ...
  comparison.png
```

## Metrics Logged

Per episode:
- reward
- moving average reward
- loss
- epsilon
- episode time

Global summary:
- average episode time
- total training time
- time to convergence (if reached)
- episodes to convergence
- fitness

## Statistical Evaluation

Generate statistical summaries from run logs:

```bash
python evaluate_stats.py --runs-dir runs
```

Optional CSV output:

```bash
python evaluate_stats.py --runs-dir runs --format csv --output stats_summary.csv
```

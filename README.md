# Parallel HPO Scalability on Multicore Hardware

This repository contains the experiment code for a CSE449 project on process-parallel hyperparameter optimization (HPO). The study uses Random Forest grid search as a fixed workload and measures how execution changes when the outer worker count increases from 1 to 2, 4, and 8.

The project is a parallel-performance study. It does not introduce a new HPO method, and it does not use GPU acceleration. Each Random Forest fit is restricted to one internal thread so that the measured concurrency comes from the outer HPO workers.

## Experiment summary

- Dataset: Forest CoverType from scikit-learn
- Estimator: `RandomForestClassifier`
- Search method: `GridSearchCV`
- Parameter combinations: 12
- Cross-validation: 5-fold stratified CV
- Fits per search: 60
- Scoring metric: Macro F1
- Outer workers: 1, 2, 4, and 8
- Process backend: Joblib Loky
- Estimator threads: 1
- Numerical-library threads per worker: 1
- Primary workload: W25, containing 25,000 training samples
- Workload comparison: W100, containing 100,000 training samples

The original experiment was run with Python 3.11.15 on an Intel Core i5-1135G7 system with 4 physical cores, 8 logical processors, and about 16 GB of RAM. Timings from another computer should not be expected to match exactly.

## Repository contents

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
`-- src/
    |-- load_dataset.py
    |-- environment_check.py
    |-- create_split.py
    |-- create_workloads.py
    |-- baseline_dummy.py
    |-- baseline_models.py
    |-- calibrate_workloads.py
    |-- calibrate_hgb_fixed.py
    |-- check_hgb_iterations.py
    |-- check_worker_threads.py
    |-- check_rf_worker_threads.py
    |-- hpo_experiment.py
    |-- sequential_hpo.py
    |-- final_hpo_single.py
    |-- run_final_benchmark.py
    |-- rq4_w100_single.py
    `-- run_rq4_w100_benchmark.py
```

The repository intentionally excludes the downloaded dataset, processed indices, benchmark outputs, report files, literature material, and scripts that generate figures, tables, references, or documents.

## Setup

Clone the repository and run all commands from its root directory.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The benchmark was developed and measured on Windows. The code uses standard Python paths and can run on other operating systems, but performance results should be treated as a new hardware and software environment.

## Prepare the dataset and fixed workloads

Run these commands in order:

```text
python src/environment_check.py
python src/load_dataset.py
python src/create_split.py
python src/create_workloads.py
```

`load_dataset.py` downloads Forest CoverType through `sklearn.datasets.fetch_covtype`, so an internet connection is needed the first time. It saves the dataset as `data/raw/covertype.csv`.

`environment_check.py` reports the thread pools visible in the normal Python environment. Those values may be greater than one. The two final single-run scripts set the supported thread environment variables before importing numerical libraries and also apply `threadpool_limits(1)` during the benchmark.

The remaining preparation scripts create:

- `data/processed/split_indices.npz`: an 80/20 stratified split using `random_state=42`
- `data/processed/workload_indices.npz`: nested W25, W100, W250, and full-training index sets

The dataset and generated indices are ignored by Git. They can be recreated from the scripts and pinned software versions.

## Run the primary W25 benchmark

The following command reproduces the primary schedule of five repetitions for each worker count:

```text
python src/run_final_benchmark.py --repetitions 5 --label final_v1
```

The harness creates a randomized block schedule using seed `4492026`. Every measurement runs in a fresh Python subprocess, and the harness waits 10 seconds between runs. Output is written under:

```text
results/raw/final_benchmark/final_v1/
```

Use a new label if that directory already contains results. Keeping separate labels prevents accidental mixing or replacement of benchmark suites.

For a short functional check before committing to the complete benchmark, use:

```text
python src/run_final_benchmark.py --repetitions 1 --label smoke_w25
```

This is still a real four-run benchmark and can take several minutes.

## Run the W100 workload comparison

The W100 suite uses three repetitions for each worker count:

```text
python src/run_rq4_w100_benchmark.py --repetitions 3 --label w100_v1
```

Its randomized schedule uses seed `4496100`, and output is written under:

```text
results/raw/rq4_w100/w100_v1/
```

W100 takes much longer than W25. Run it when the computer can remain connected to power and mostly idle.

## Independent reproduction runs

Independent validation suites should use new labels and must remain separate from the primary results:

```text
python src/run_final_benchmark.py --repetitions 5 --label reproduction_w25
python src/run_rq4_w100_benchmark.py --repetitions 3 --label reproduction_w100
```

Do not combine these reruns with the primary measurements to increase the reported sample size. They are separate reproducibility evidence.

## Optional pilot and calibration scripts

These scripts document the earlier workload selection and thread-control checks. They are not required to launch the two final benchmark harnesses after the dataset and workload files have been prepared.

```text
python src/baseline_dummy.py
python src/baseline_models.py
python src/calibrate_workloads.py
python src/check_hgb_iterations.py
python src/calibrate_hgb_fixed.py
python src/check_worker_threads.py
python src/check_rf_worker_threads.py
python src/sequential_hpo.py
python src/hpo_experiment.py --workers 1
```

The baseline scripts use the fixed test partition for early model characterization. Final HPO selection uses only the training partition and cross-validation.

## Output files

Each final benchmark directory contains:

- `schedule.csv`: randomized execution order
- `all_runs.csv`: one row per completed run
- `summary_by_workers.csv`: mean runtime, sample standard deviation, 95% confidence interval, speedup, efficiency, throughput, CPU, and memory summaries
- `runs/*.json`: raw metrics for individual runs
- `runs/*_candidates.csv`: GridSearchCV candidate results

Speedup is calculated from the worker-group means:

```text
speedup(p) = mean_runtime(1 worker) / mean_runtime(p workers)
```

Efficiency is:

```text
efficiency(p) = speedup(p) / p
```

## Reference results

The original W25 benchmark produced the following mean search runtimes:

| Workers | Runtime (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 126.299712 | 1.000000 | 100.00% |
| 2 | 74.968672 | 1.684700 | 84.23% |
| 4 | 49.761834 | 2.538084 | 63.45% |
| 8 | 43.150796 | 2.926938 | 36.59% |

The W100 benchmark produced speedups of 1.647361, 2.341547, and 2.667225 at 2, 4, and 8 workers. W100 required about 4.36 times more sequential computation than W25 but scaled slightly worse in the tested environment.

Within each workload, every primary run selected the same best hyperparameters and Macro F1 score. W25 selected 100 trees, no depth limit, and `min_samples_leaf=1`, with a best CV Macro F1 of 0.746271. W100 selected 200 trees, no depth limit, and `min_samples_leaf=1`, with a score of 0.843836.

These values are reference points, not universal performance targets. Background processes, power settings, thermal conditions, operating system, library builds, and hardware can change timing and resource measurements.

## Reproducibility controls

The final benchmark code applies the following controls:

- fixed dataset split and workload seeds
- fixed five-fold stratified CV
- fixed Random Forest parameter grid
- `RandomForestClassifier(n_jobs=1)`
- Joblib Loky process workers
- one inner numerical-library thread per worker
- fresh subprocess for each timed run
- randomized blocked run order
- 10-second cooldown between runs
- separate search and manual refit timing
- 250 ms process-tree resource sampling
- correctness checks across worker counts

## Important interpretation notes

- The GPU is not used.
- The original CPU has 4 physical cores and 8 logical processors.
- Aggregate process-tree RSS is an approximate process-memory indicator. Shared pages can be counted more than once, so it is not exact unique physical RAM usage.
- The test partition was used during early pilot baseline characterization. Final HPO and hyperparameter selection use only the training partition and cross-validation.
- W25 and W100 are separate benchmark suites and should not be pooled.
- Independent reproduction runs are validation evidence and should not be pooled with the primary runs.
- Eight workers gave the lowest tested runtime, while four workers retained greater efficiency. Neither worker count should be treated as universally optimal.
- The measured results apply to the tested machine and workload. They do not automatically generalize to other CPUs, datasets, estimators, or HPO methods.

## Clean repository policy

All code uses paths relative to the repository root. No user-specific path, account name, credential, API key, or private dataset is required. The `.gitignore` prevents local environments, downloaded data, generated results, logs, IDE settings, and common secret files from being committed accidentally.

The experiment scripts in this repository match the frozen experiment source recorded in commit `b30c645fd9f213769f64abbcba4de272237c8463` of the original course-project repository.


import os

# Set numerical-library thread limits before importing
# NumPy / scikit-learn.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import argparse
import json
import threading
import time

import numpy as np
import pandas as pd
import psutil

from joblib import parallel_config

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
)

from threadpoolctl import threadpool_limits


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "covertype.csv"
)

WORKLOAD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "workload_indices.npz"
)

TARGET_COLUMN = "Cover_Type"

RANDOM_STATE = 42
CV_FOLDS = 5

# Resource sampling interval in seconds.
MONITOR_INTERVAL = 0.25


PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, None],
    "min_samples_leaf": [1, 4],
}


# ============================================================
# RESOURCE MONITOR
# ============================================================

class ResourceMonitor:
    def __init__(
        self,
        pid,
        interval=0.25,
    ):
        self.root = psutil.Process(pid)
        self.interval = interval

        self.stop_event = threading.Event()
        self.thread = None

        # Keep the same psutil.Process objects between samples.
        # cpu_percent(interval=None) requires previous measurements
        # from the same Process instance.
        self.process_cache = {}

        self.tree_cpu_raw = []
        self.tree_cpu_normalized = []
        self.tree_rss_mb = []

        self.system_cpu = []
        self.system_ram = []

        self.logical_cpus = (
            psutil.cpu_count(logical=True)
            or 1
        )

    def _discover_processes(self):
        discovered = {}

        try:
            discovered[self.root.pid] = self.root

            for child in self.root.children(
                recursive=True
            ):
                discovered[child.pid] = child

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            pass

        # Add newly discovered processes to the persistent cache.
        # Initialize cpu_percent once so future measurements
        # represent CPU usage over an interval.
        for pid, process in discovered.items():

            if pid not in self.process_cache:

                self.process_cache[pid] = process

                try:
                    process.cpu_percent(
                        interval=None
                    )

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                ):
                    pass

        # Remove child processes that no longer exist.
        active_pids = set(
            discovered.keys()
        )

        for pid in list(
            self.process_cache.keys()
        ):
            if pid not in active_pids:
                self.process_cache.pop(
                    pid,
                    None,
                )

    def _sample(self):
        self._discover_processes()

        total_cpu = 0.0
        total_rss = 0

        for process in list(
            self.process_cache.values()
        ):
            try:
                total_cpu += (
                    process.cpu_percent(
                        interval=None
                    )
                )

                total_rss += (
                    process.memory_info().rss
                )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):
                continue

        # psutil process CPU can exceed 100% when multiple
        # processes/cores are active.
        #
        # Example on an 8-logical-CPU machine:
        # raw 400% / 8 = 50% total-machine capacity.
        normalized_cpu = (
            total_cpu
            / self.logical_cpus
        )

        self.tree_cpu_raw.append(
            total_cpu
        )

        self.tree_cpu_normalized.append(
            normalized_cpu
        )

        self.tree_rss_mb.append(
            total_rss
            / (1024 ** 2)
        )

        self.system_cpu.append(
            psutil.cpu_percent(
                interval=None
            )
        )

        self.system_ram.append(
            psutil.virtual_memory().percent
        )

    def _run(self):
        # Initialize system-wide CPU measurement.
        psutil.cpu_percent(
            interval=None
        )

        # Initialize parent/known process counters.
        self._discover_processes()

        while not self.stop_event.wait(
            self.interval
        ):
            self._sample()

    def start(self):
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

        # Take one final reading.
        self._sample()

    @staticmethod
    def safe_mean(values):
        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    @staticmethod
    def safe_max(values):
        if not values:
            return 0.0

        return float(
            np.max(values)
        )

    def summary(self):
        return {
            "resource_samples":
                len(self.tree_cpu_raw),

            "mean_process_tree_cpu_raw_percent":
                self.safe_mean(
                    self.tree_cpu_raw
                ),

            "peak_process_tree_cpu_raw_percent":
                self.safe_max(
                    self.tree_cpu_raw
                ),

            "mean_process_tree_cpu_normalized_percent":
                self.safe_mean(
                    self.tree_cpu_normalized
                ),

            "peak_process_tree_cpu_normalized_percent":
                self.safe_max(
                    self.tree_cpu_normalized
                ),

            "mean_process_tree_rss_mb":
                self.safe_mean(
                    self.tree_rss_mb
                ),

            "peak_process_tree_rss_mb":
                self.safe_max(
                    self.tree_rss_mb
                ),

            "mean_system_cpu_percent":
                self.safe_mean(
                    self.system_cpu
                ),

            "peak_system_cpu_percent":
                self.safe_max(
                    self.system_cpu
                ),

            "mean_system_ram_percent":
                self.safe_mean(
                    self.system_ram
                ),

            "peak_system_ram_percent":
                self.safe_max(
                    self.system_ram
                ),
        }


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run one controlled Random Forest "
            "parallel-HPO benchmark."
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        choices=[1, 2, 4, 8],
        help="Number of outer HPO workers.",
    )

    parser.add_argument(
        "--repetition",
        type=int,
        required=True,
        help="Benchmark repetition number.",
    )

    parser.add_argument(
        "--run-position",
        type=int,
        required=True,
        help="Position of this run in randomized schedule.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where raw benchmark results are saved.",
    )

    return parser.parse_args()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    args = parse_arguments()

    workers = args.workers
    repetition = args.repetition
    run_position = args.run_position

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)

    print(
        f"FINAL HPO RUN | "
        f"workers={workers} | "
        f"repetition={repetition}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_FILE
    )

    workloads = np.load(
        WORKLOAD_FILE
    )

    # Primary benchmark workload:
    # nested W25 training subset.
    train_indices = workloads[
        "train_25000"
    ]

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[
        TARGET_COLUMN
    ]

    X_train = X.iloc[
        train_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    # --------------------------------------------------------
    # SEARCH SIZE
    # --------------------------------------------------------

    candidate_count = len(
        list(
            ParameterGrid(
                PARAM_GRID
            )
        )
    )

    total_cv_fits = (
        candidate_count
        * CV_FOLDS
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # GRID SEARCH
    #
    # IMPORTANT:
    # refit=False allows exact measurement of the parallel
    # search separately from the final sequential refit.
    # --------------------------------------------------------

    search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=workers,
        pre_dispatch="n_jobs",
        refit=False,
        return_train_score=False,
        error_score="raise",
        verbose=0,
    )

    print(
        f"Starting {total_cv_fits}-fit search "
        f"with {workers} worker(s)..."
    )

    # --------------------------------------------------------
    # RESOURCE MONITOR
    #
    # This measures the HPO SEARCH.
    # Manual final refit is measured separately.
    # --------------------------------------------------------

    monitor = ResourceMonitor(
        pid=os.getpid(),
        interval=MONITOR_INTERVAL,
    )

    monitor.start()

    # --------------------------------------------------------
    # PARALLEL HPO SEARCH
    # --------------------------------------------------------

    with threadpool_limits(
        limits=1
    ):
        with parallel_config(
            backend="loky",
            inner_max_num_threads=1,
        ):

            search_start = (
                time.perf_counter()
            )

            search.fit(
                X_train,
                y_train,
            )

            search_runtime = (
                time.perf_counter()
                - search_start
            )

    monitor.stop()

    resources = (
        monitor.summary()
    )

    # --------------------------------------------------------
    # SEARCH RESULT
    # --------------------------------------------------------

    best_parameters = (
        search.best_params_
    )

    best_cv_macro_f1 = float(
        search.best_score_
    )

    print(
        "Search complete."
    )

    print(
        f"Search runtime: "
        f"{search_runtime:.4f} s"
    )

    print(
        f"Best Macro F1: "
        f"{best_cv_macro_f1:.6f}"
    )

    print(
        "Best parameters:"
    )

    for key, value in (
        best_parameters.items()
    ):
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # MANUAL REFIT
    #
    # The best configuration is trained exactly once
    # using all W25 training samples.
    #
    # This stays single-threaded.
    # --------------------------------------------------------

    best_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        **best_parameters,
    )

    print(
        "Starting manual refit..."
    )

    with threadpool_limits(
        limits=1
    ):

        refit_start = (
            time.perf_counter()
        )

        best_model.fit(
            X_train,
            y_train,
        )

        refit_runtime = (
            time.perf_counter()
            - refit_start
        )

    # --------------------------------------------------------
    # FINAL PERFORMANCE METRICS
    # --------------------------------------------------------

    total_runtime = (
        search_runtime
        + refit_runtime
    )

    throughput = (
        total_cv_fits
        / search_runtime
    )

    # --------------------------------------------------------
    # CANDIDATE-LEVEL CV RESULTS
    # --------------------------------------------------------

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    candidate_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
    ]

    candidate_results = (
        cv_results[
            candidate_columns
        ]
        .sort_values(
            "rank_test_score"
        )
    )

    # --------------------------------------------------------
    # OUTPUT FILE NAME
    # --------------------------------------------------------

    stem = (
        f"run_{run_position:02d}"
        f"_rep_{repetition}"
        f"_workers_{workers}"
    )

    candidate_file = (
        output_dir
        / f"{stem}_candidates.csv"
    )

    candidate_results.to_csv(
        candidate_file,
        index=False,
    )

    # --------------------------------------------------------
    # RAW RUN RESULT
    # --------------------------------------------------------

    result = {
        "run_position":
            run_position,

        "repetition":
            repetition,

        "workers":
            workers,

        "workload":
            "W25",

        "training_samples":
            int(
                len(X_train)
            ),

        "candidate_count":
            candidate_count,

        "cv_folds":
            CV_FOLDS,

        "total_cv_fits":
            total_cv_fits,

        "estimator_n_jobs":
            1,

        "inner_threads":
            1,

        "search_runtime_seconds":
            search_runtime,

        "refit_runtime_seconds":
            refit_runtime,

        "total_runtime_seconds":
            total_runtime,

        "throughput_fits_per_second":
            throughput,

        "best_cv_macro_f1":
            best_cv_macro_f1,

        "best_parameters":
            best_parameters,

        "random_state":
            RANDOM_STATE,

        **resources,
    }

    result_file = (
        output_dir
        / f"{stem}.json"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print(
        f"Refit runtime: "
        f"{refit_runtime:.4f} s"
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.4f} s"
    )

    print(
        f"Throughput: "
        f"{throughput:.4f} fits/s"
    )

    print(
        "Mean raw process-tree CPU: "
        f"{resources['mean_process_tree_cpu_raw_percent']:.2f}%"
    )

    print(
        "Mean normalized process CPU: "
        f"{resources['mean_process_tree_cpu_normalized_percent']:.2f}%"
    )

    print(
        "Peak normalized process CPU: "
        f"{resources['peak_process_tree_cpu_normalized_percent']:.2f}%"
    )

    print(
        "Mean system CPU: "
        f"{resources['mean_system_cpu_percent']:.2f}%"
    )

    print(
        "Peak process-tree RSS: "
        f"{resources['peak_process_tree_rss_mb']:.2f} MB"
    )

    print(
        "Resource samples: "
        f"{resources['resource_samples']}"
    )

    print(
        "Saved:"
    )

    print(
        result_file
    )

    print(
        candidate_file
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
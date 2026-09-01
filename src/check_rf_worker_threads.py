import os

from joblib import (
    Parallel,
    delayed,
    parallel_config,
)

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from threadpoolctl import threadpool_info


def inspect_worker(task_id):
    # Create a small local workload inside
    # the child process.
    X, y = make_classification(
        n_samples=5_000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=7,
        random_state=42 + task_id,
    )

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X, y)

    pools = []

    for pool in threadpool_info():
        pools.append(
            {
                "internal_api": pool.get(
                    "internal_api"
                ),
                "num_threads": pool.get(
                    "num_threads"
                ),
                "prefix": pool.get(
                    "prefix"
                ),
            }
        )

    return {
        "task_id": task_id,
        "pid": os.getpid(),
        "pools": pools,
    }


def main():
    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
    ):
        results = Parallel(
            n_jobs=2
        )(
            delayed(inspect_worker)(i)
            for i in range(4)
        )

    workers = {}

    for result in results:
        workers[result["pid"]] = (
            result["pools"]
        )

    print(
        "Unique worker processes:",
        len(workers),
    )

    for pid, pools in workers.items():

        print(
            f"\nWorker PID: {pid}"
        )

        for pool in pools:
            print(
                f"{pool['internal_api']} "
                f"| threads = "
                f"{pool['num_threads']} "
                f"| prefix = "
                f"{pool['prefix']}"
            )


if __name__ == "__main__":
    main()
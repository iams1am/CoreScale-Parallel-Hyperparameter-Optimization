import os
import time

from joblib import (
    Parallel,
    delayed,
    parallel_config,
)
from threadpoolctl import threadpool_info


def inspect_worker(task_id):
    # Keep tasks alive briefly so both workers
    # are likely to participate.
    time.sleep(1)

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
                f"{pool['num_threads']}"
            )


if __name__ == "__main__":
    main()
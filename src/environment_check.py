import os
import platform

import numpy
import pandas
import psutil
import scipy
import sklearn
from threadpoolctl import threadpool_info


def main():
    print("=== CSE449 Parallel HPO Environment Check ===")

    print("\nSYSTEM")
    print("Operating system:", platform.platform())
    print("Logical processors:", os.cpu_count())
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Logical cores:", psutil.cpu_count(logical=True))

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"RAM: {ram_gb:.2f} GB")

    print("\nSOFTWARE")
    print("Python:", platform.python_version())
    print("NumPy:", numpy.__version__)
    print("Pandas:", pandas.__version__)
    print("SciPy:", scipy.__version__)
    print("Scikit-learn:", sklearn.__version__)

    print("\nTHREAD POOLS")
    for pool in threadpool_info():
        print(
            f"{pool.get('internal_api')} | "
            f"threads={pool.get('num_threads')} | "
            f"version={pool.get('version')}"
        )


if __name__ == "__main__":
    main()
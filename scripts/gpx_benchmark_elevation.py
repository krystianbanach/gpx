import json
import time

from gpx_parser import parser_py, parser_np

from gpx_list import elevation_gain_loss
from gpx_numpy import elevation_gain_loss_np
from gpx_numba import elevation_gain_loss_numba


FILES = [
    "data/bieg_10000_1s.gpx",
    "data/bieg_25000_1s.gpx",
    "data/bieg_50000_1s.gpx",
    "data/bieg_100000_1s.gpx",
    "data/bieg_500000_1s.gpx",
    "data/bieg_1000000_1s.gpx",
]

OUTPUT_FILE = "results/benchmark_elevation2.json"

WARMUP_RUNS = 3
MEASURED_RUNS = 10


def measure_once(function, *args):
    start = time.perf_counter()
    function(*args)
    return time.perf_counter() - start


def measure_repeated(function, *args):
    for _ in range(WARMUP_RUNS):
        function(*args)

    times = []

    for _ in range(MEASURED_RUNS):
        elapsed_time = measure_once(function, *args)
        times.append(elapsed_time)

    return {
        "czas": sum(times) / len(times),
        "czasy": times,
    }


def benchmark(files, output_file=OUTPUT_FILE):
    results = []

    for file in files:
        print("Pomiar przewyższeń dla pliku:", file)

        _, _, ele, _ = parser_py(file)
        _, _, ele_np, _ = parser_np(file)

        list_result = measure_repeated(
            elevation_gain_loss,
            ele,
        )

        numpy_result = measure_repeated(
            elevation_gain_loss_np,
            ele_np,
        )

        numba_result = measure_repeated(
            elevation_gain_loss_numba,
            ele_np,
        )

        record = {
            "plik": file,
            "liczba_punktow": len(ele),
            "operacja": "przewyzszenia",
            "liczba_warmupow": WARMUP_RUNS,
            "liczba_pomiarow": MEASURED_RUNS,
            "sredni_czas_lista": list_result["czas"],
            "sredni_czas_numpy": numpy_result["czas"],
            "sredni_czas_numba": numba_result["czas"],
            "czasy_lista": list_result["czasy"],
            "czasy_numpy": numpy_result["czasy"],
            "czasy_numba": numba_result["czasy"],
        }

        results.append(record)

        print("  czas_lista:", record["sredni_czas_lista"])
        print("  czas_numpy:", record["sredni_czas_numpy"])
        print("  czas_numba:", record["sredni_czas_numba"])

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Zapisano wyniki do:", output_file)


if __name__ == "__main__":
    benchmark(FILES)
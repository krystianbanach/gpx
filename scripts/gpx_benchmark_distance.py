import json
import time

from gpx_parser import parser_py, parser_np
from gpx_list import haversine_total_distance
from gpx_numpy import haversine_total_distance_np
from gpx_numba import haversine_total_distance_numba

FILES = [
    "data/trasa_10000.gpx",
    "data/trasa_25000.gpx",
    "data/trasa_50000.gpx",
    "data/trasa_100000.gpx",
    "data/trasa_250000.gpx",
    "data/trasa_500000.gpx",
    "data/trasa_1000000.gpx",
]

OUTPUT_FILE = "results/benchmark_distance.json"

WARMUP_RUNS = 3
MEASURED_RUNS = 10


def measure(function, *args):
    for _ in range(WARMUP_RUNS):
        function(*args)

    times = []
    last_result = None

    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        last_result = function(*args)
        elapsed_time = time.perf_counter() - start

        times.append(elapsed_time)

    return {
        "czas": sum(times) / len(times),
        "czasy": times,
        "wynik": last_result,
    }


def benchmark(files, output_file=OUTPUT_FILE):
    results = []

    for file in files:
        print("Pomiar dystansu dla pliku:", file)

        lat, lon, _ = parser_py(file)
        lat_np, lon_np, _ = parser_np(file)

        list_result = measure(
            haversine_total_distance,
            lat,
            lon,
        )

        numpy_result = measure(
            haversine_total_distance_np,
            lat_np,
            lon_np,
        )

        numba_result = measure(
            haversine_total_distance_numba,
            lat_np,
            lon_np,
        )

        record = {
            "plik": file,
            "liczba_punktow": len(lat),
            "operacja": "dystans",
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
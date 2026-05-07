import json
import time

from gpx_parser import parser_py, parser_np
from gpx_list import haversine_total_distance
from gpx_numpy import haversine_total_distance_np


FILES = [
    "bieg_10000_1s.gpx",
    "bieg_25000_1s.gpx",
    "bieg_50000_1s.gpx",
    "bieg_100000_1s.gpx",
    "bieg_500000_1s.gpx",
    "bieg_1000000_1s.gpx",
]

OUTPUT_FILE = "benchmark_distance.json"
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
        times.append(measure_once(function, *args))

    return {
        "czas": sum(times) / len(times),
        "minimalny_czas": min(times),
        "maksymalny_czas": max(times),
        "czasy": times,
    }


def benchmark(files, output_file=OUTPUT_FILE):
    results = []

    for file in files:
        print("Pomiar dystansu dla pliku:", file)

        lat, lon, ele, times = parser_py(file)
        lat_np, lon_np, ele_np, times_np = parser_np(file)

        list_result = measure_repeated(haversine_total_distance, lat, lon)
        numpy_result = measure_repeated(haversine_total_distance_np, lat_np, lon_np)

        speedup = list_result["czas"] / numpy_result["czas"] if numpy_result["czas"] != 0 else None

        record = {
            "plik": file,
            "liczba_punktow": len(lat),
            "operacja": "dystans",
            "liczba_warmupow": WARMUP_RUNS,
            "liczba_pomiarow": MEASURED_RUNS,
            "czas_lista": list_result["czas"],
            "czas_numpy": numpy_result["czas"],
            "minimalny_czas_lista": list_result["minimalny_czas"],
            "minimalny_czas_numpy": numpy_result["minimalny_czas"],
            "maksymalny_czas_lista": list_result["maksymalny_czas"],
            "maksymalny_czas_numpy": numpy_result["maksymalny_czas"],
            "przyspieszenie": speedup,
            "czasy_lista": list_result["czasy"],
            "czasy_numpy": numpy_result["czasy"],
        }

        results.append(record)

        print("  czas_lista:", record["czas_lista"])
        print("  czas_numpy:", record["czas_numpy"])
        print("  przyspieszenie:", record["przyspieszenie"])

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Zapisano wyniki do:", output_file)


if __name__ == "__main__":
    benchmark(FILES)

import json
import time

from gpx_parser import parser_py, parser_np

from gpx_list import frechet_distance
from gpx_numpy import frechet_distance_np
from gpx_numba import frechet_distance_numba


ROUTE_PAIRS = [
    ("data/bieg_1000.gpx","data/bieg_1000_zawodnik.gpx"),
    ("data/bieg_2000.gpx","data/bieg_2000_zawodnik.gpx"),
    ("data/bieg_5000.gpx","data/bieg_5000_zawodnik.gpx"),
    ("data/bieg_10000_1s.gpx","data/bieg_10000_1s_zawodnik.gpx"),
    #("data/bieg_25000_1s.gpx","data/bieg_25000_1s_zawodnik.gpx"),
    #("data/bieg_50000_1s.gpx","data/bieg_50000_1s_zawodnik.gpx"),
]

OUTPUT_FILE = "results/benchmark_frechet_distance.json"

WARMUP_RUNS = 0
MEASURED_RUNS = 2


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


def benchmark(route_pairs, output_file=OUTPUT_FILE):
    results = []

    for reference_file, runner_file in route_pairs:
        print("Pomiar Fréchet distance:")
        print("  referencja:", reference_file)
        print("  zawodnik:  ", runner_file)

        ref_lat, ref_lon, _, _ = parser_py(reference_file)
        run_lat, run_lon, _, _ = parser_py(runner_file)

        ref_lat_np, ref_lon_np, _, _ = parser_np(reference_file)
        run_lat_np, run_lon_np, _, _ = parser_np(runner_file)

        reference_points = len(ref_lat)
        runner_points = len(run_lat)

        # Warmup Numby.
        # To wywołanie kompiluje funkcje @njit na małym fragmencie danych.
        frechet_distance_numba(
            ref_lat_np[:10],
            ref_lon_np[:10],
            run_lat_np[:10],
            run_lon_np[:10],
        )

        list_result = measure_repeated(
            frechet_distance,
            ref_lat,
            ref_lon,
            run_lat,
            run_lon,
        )

        numpy_result = measure_repeated(
            frechet_distance_np,
            ref_lat_np,
            ref_lon_np,
            run_lat_np,
            run_lon_np,
        )

        numba_result = measure_repeated(
            frechet_distance_numba,
            ref_lat_np,
            ref_lon_np,
            run_lat_np,
            run_lon_np,
        )

        record = {
            "trasa_referencyjna": reference_file,
            "trasa_zawodnika": runner_file,
            "liczba_punktow_referencji": reference_points,
            "liczba_punktow_zawodnika": runner_points,
            "liczba_komorek_dp": reference_points * runner_points,
            "operacja": "frechet_distance",
            "liczba_warmupow": WARMUP_RUNS,
            "liczba_pomiarow": MEASURED_RUNS,
            "warmup_numba": True,
            "sredni_czas_lista": list_result["czas"],
            "sredni_czas_numpy": numpy_result["czas"],
            "sredni_czas_numba": numba_result["czas"],
            "czasy_lista": list_result["czasy"],
            "czasy_numpy": numpy_result["czasy"],
            "czasy_numba": numba_result["czasy"],
        }

        results.append(record)

        print("  liczba punktów referencji:", reference_points)
        print("  liczba punktów zawodnika:", runner_points)
        print("  liczba komórek DP:", reference_points * runner_points)
        print("  sredni_czas_lista:", record["sredni_czas_lista"])
        print("  sredni_czas_numpy:", record["sredni_czas_numpy"])
        print("  sredni_czas_numba:", record["sredni_czas_numba"])
        print()

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Zapisano wyniki do:", output_file)


if __name__ == "__main__":
    benchmark(ROUTE_PAIRS)
import json
import time

from gpx_parser import parser_py, parser_np

from gpx_list import compare_routes
from gpx_numpy import compare_routes_np
from gpx_numba import compare_routes_numba


ROUTE_PAIRS = [
    ("data/trasa_1000.gpx","data/trasa_1000_zawodnik.gpx"),
    ("data/trasa_2000.gpx","data/trasa_2000_zawodnik.gpx"),
    ("data/trasa_3000.gpx","data/trasa_3000_zawodnik.gpx"),
    ("data/trasa_4000.gpx","data/trasa_4000_zawodnik.gpx"),
    ("data/trasa_5000.gpx","data/trasa_5000_zawodnik.gpx"),
    ("data/trasa_6000.gpx","data/trasa_6000_zawodnik.gpx"),
    ("data/trasa_7000.gpx","data/trasa_7000_zawodnik.gpx"),
    ("data/trasa_8000.gpx","data/trasa_8000_zawodnik.gpx"),
    ("data/trasa_9000.gpx","data/trasa_9000_zawodnik.gpx"),
    ("data/trasa_10000.gpx","data/trasa_10000_zawodnik.gpx"),
]

OUTPUT_FILE = "results/benchmark_route_comparison.json"

WARMUP_RUNS = 2
MEASURED_RUNS = 5


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


def benchmark(route_pairs, output_file=OUTPUT_FILE):
    results = []

    for reference_file, runner_file in route_pairs:
        print("Pomiar porównania tras:")
        print("  referencja:", reference_file)
        print("  zawodnik:  ", runner_file)

        ref_lat, ref_lon, _ = parser_py(reference_file)
        run_lat, run_lon, _ = parser_py(runner_file)

        ref_lat_np, ref_lon_np, _ = parser_np(reference_file)
        run_lat_np, run_lon_np, _ = parser_np(runner_file)

        reference_points = len(ref_lat)
        runner_points = len(run_lat)

        # Warmup Numby.
        # To nie jest właściwy pomiar — chodzi tylko o kompilację funkcji @njit.
        compare_routes_numba(
            ref_lat_np[:10],
            ref_lon_np[:10],
            run_lat_np[:10],
            run_lon_np[:10],
        )

        list_result = measure(
            compare_routes,
            ref_lat,
            ref_lon,
            run_lat,
            run_lon,
        )

        numpy_result = measure(
            compare_routes_np,
            ref_lat_np,
            ref_lon_np,
            run_lat_np,
            run_lon_np,
        )

        numba_result = measure(
            compare_routes_numba,
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
            "liczba_sprawdzen_punkt_segment": runner_points * (reference_points - 1),
            "operacja": "porownanie_tras",
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

        print("  sredni_czas_lista:", record["sredni_czas_lista"])
        print("  sredni_czas_numpy:", record["sredni_czas_numpy"])
        print("  sredni_czas_numba:", record["sredni_czas_numba"])
        print()

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Zapisano wyniki do:", output_file)


if __name__ == "__main__":
    benchmark(ROUTE_PAIRS)
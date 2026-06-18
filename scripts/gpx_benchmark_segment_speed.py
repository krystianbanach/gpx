import json
import time
from pathlib import Path

from gpx_parser import parser_py, parser_np

from gpx_list import compare_segment_speeds
from gpx_numpy import compare_segment_speeds_np
from gpx_numba import compare_segment_speeds_numba


ROUTE_PAIRS = [
    ("data/trasa_10000.gpx", "data/trasa_10000_zawodnik.gpx"),
    ("data/trasa_25000.gpx", "data/trasa_25000_zawodnik.gpx"),
    ("data/trasa_50000.gpx", "data/trasa_50000_zawodnik.gpx"),
    ("data/trasa_100000.gpx", "data/trasa_100000_zawodnik.gpx"),
    ("data/trasa_250000.gpx", "data/trasa_250000_zawodnik.gpx"),
    ("data/trasa_500000.gpx", "data/trasa_500000_zawodnik.gpx"),
    ("data/trasa_1000000.gpx", "data/trasa_1000000_zawodnik.gpx"),
]

OUTPUT_FILE = "results/benchmark_segment_speed.json"

SEGMENT_LENGTH_M = 500.0

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


def benchmark(route_pairs, output_file=OUTPUT_FILE):
    results = []

    for reference_file, runner_file in route_pairs:
        print("Pomiar segmentowego porównania prędkości:")
        print("  referencja:", reference_file)
        print("  zawodnik:  ", runner_file)

        ref_lat, ref_lon, _, ref_time = parser_py(reference_file)
        run_lat, run_lon, _, run_time = parser_py(runner_file)

        ref_lat_np, ref_lon_np, _, ref_time_np = parser_np(reference_file)
        run_lat_np, run_lon_np, _, run_time_np = parser_np(runner_file)

        compare_segment_speeds_numba(
            ref_lat_np[:2000],
            ref_lon_np[:2000],
            ref_time_np[:2000],
            run_lat_np[:2000],
            run_lon_np[:2000],
            run_time_np[:2000],
            SEGMENT_LENGTH_M,
        )

        list_result = measure(
            compare_segment_speeds,
            ref_lat,
            ref_lon,
            ref_time,
            run_lat,
            run_lon,
            run_time,
            SEGMENT_LENGTH_M,
        )

        numpy_result = measure(
            compare_segment_speeds_np,
            ref_lat_np,
            ref_lon_np,
            ref_time_np,
            run_lat_np,
            run_lon_np,
            run_time_np,
            SEGMENT_LENGTH_M,
        )

        numba_result = measure(
            compare_segment_speeds_numba,
            ref_lat_np,
            ref_lon_np,
            ref_time_np,
            run_lat_np,
            run_lon_np,
            run_time_np,
            SEGMENT_LENGTH_M,
        )

        record = {
            "trasa_referencyjna": reference_file,
            "trasa_zawodnika": runner_file,
            "liczba_punktow_referencji": len(ref_lat),
            "liczba_punktow_zawodnika": len(run_lat),
            "dlugosc_segmentu_m": SEGMENT_LENGTH_M,
            "operacja": "segmentowe_porownanie_predkosci",
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

        print("  czas_lista:", record["sredni_czas_lista"])
        print("  czas_numpy:", record["sredni_czas_numpy"])
        print("  czas_numba:", record["sredni_czas_numba"])
        print()

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Zapisano wyniki do:", output_file)


if __name__ == "__main__":
    benchmark(ROUTE_PAIRS)
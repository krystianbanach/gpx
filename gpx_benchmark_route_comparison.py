import json
import time

from gpx_parser import parser_py, parser_np
from gpx_list import compare_routes
from gpx_numpy import compare_routes_np


ROUTE_PAIRS = [
    ("bieg_10000_1s.gpx", "bieg_10000_1s_zawodnik.gpx"),
    #("bieg_25000_1s.gpx", "bieg_25000_1s_zawodnik.gpx"),
    #("bieg_50000_1s.gpx", "bieg_50000_1s_zawodnik.gpx"),
    # ("bieg_100000_1s.gpx", "bieg_100000_1s_zawodnik.gpx"),
]

OUTPUT_FILE = "benchmark_route_comparison2.json"
WARMUP_RUNS = 0
MEASURED_RUNS = 1


def measure_once(function, *args):
    start = time.perf_counter()
    function(*args)
    return time.perf_counter() - start


def benchmark(route_pairs, output_file=OUTPUT_FILE):
    results = []

    for reference_file, runner_file in route_pairs:
        print("Pomiar porównania tras:")
        print("  referencja:", reference_file)
        print("  zawodnik:  ", runner_file)

        ref_lat, ref_lon, ref_ele, ref_times = parser_py(reference_file)
        run_lat, run_lon, run_ele, run_times = parser_py(runner_file)

        ref_lat_np, ref_lon_np, ref_ele_np, ref_times_np = parser_np(reference_file)
        run_lat_np, run_lon_np, run_ele_np, run_times_np = parser_np(runner_file)

        list_time = measure_once(compare_routes, ref_lat, ref_lon, run_lat, run_lon)
        numpy_time = measure_once(compare_routes_np, ref_lat_np, ref_lon_np, run_lat_np, run_lon_np)

        speedup = list_time / numpy_time if numpy_time != 0 else None

        record = {
            "plik": reference_file,
            "trasa_referencyjna": reference_file,
            "trasa_zawodnika": runner_file,
            "liczba_punktow": len(ref_lat),
            "liczba_punktow_referencji": len(ref_lat),
            "liczba_punktow_zawodnika": len(run_lat),
            "liczba_sprawdzen_punkt_segment": len(run_lat) * (len(ref_lat) - 1),
            "operacja": "porownanie_tras",
            "liczba_warmupow": WARMUP_RUNS,
            "liczba_pomiarow": MEASURED_RUNS,
            "czas_lista": list_time,
            "czas_numpy": numpy_time,
            "minimalny_czas_lista": list_time,
            "minimalny_czas_numpy": numpy_time,
            "maksymalny_czas_lista": list_time,
            "maksymalny_czas_numpy": numpy_time,
            "przyspieszenie": speedup,
        }

        results.append(record)

        print("  czas_lista:", list_time)
        print("  czas_numpy:", numpy_time)
        print("  przyspieszenie:", speedup)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Zapisano wyniki do:", output_file)


if __name__ == "__main__":
    benchmark(ROUTE_PAIRS)

import os
from time import perf_counter

from gpx_parser import load_and_validate_gpx
from gpx_list import (
    total_distance_haversine_list,
    elevation_gain_loss_segments,
    compare_to_reference_route_list,
)
from gpx_numpy import (
    segments_to_columns,
    total_distance_segments_numpy,
    elevation_gain_loss_segments_numpy,
    compare_to_reference_route_numpy,
)

RUNNER_GPX_FILE = "bieg_10000_1s.gpx"
REFERENCE_GPX_FILE = "bieg_10000_1s_zawodnik.gpx"

REPEATS = 10
WARMUPS = 3

REFERENCE_REPEATS = 1
REFERENCE_WARMUPS = 0


def total_distance_segments_list(route_segments):
    total_distance = 0.0
    for segment in route_segments:
        if len(segment) >= 2:
            total_distance += total_distance_haversine_list(segment)
    return total_distance


def measure_time(func, repeats, warmups):
    for _ in range(warmups):
        func()

    times = []
    for _ in range(repeats):
        start = perf_counter()
        func()
        end = perf_counter()
        times.append((end - start) * 1000.0)

    avg = sum(times) / len(times)
    return avg, min(times), max(times)


def main():
    runner_segments, _ = load_and_validate_gpx(RUNNER_GPX_FILE)
    reference_segments, _ = load_and_validate_gpx(REFERENCE_GPX_FILE)

    runner_segments_numpy = segments_to_columns(runner_segments)

    distance_list_result_m = total_distance_segments_list(runner_segments)
    distance_numpy_result_m = total_distance_segments_numpy(runner_segments_numpy)

    gain_list, loss_list = elevation_gain_loss_segments(runner_segments)
    gain_numpy, loss_numpy = elevation_gain_loss_segments_numpy(runner_segments_numpy)

    reference_list_result = compare_to_reference_route_list(
        reference_segments,
        runner_segments,
    )
    reference_numpy_result = compare_to_reference_route_numpy(
        reference_segments,
        runner_segments,
    )

    dist_list_avg, dist_list_min, dist_list_max = measure_time(
        lambda: total_distance_segments_list(runner_segments),
        REPEATS,
        WARMUPS,
    )

    dist_numpy_avg, dist_numpy_min, dist_numpy_max = measure_time(
        lambda: total_distance_segments_numpy(runner_segments_numpy),
        REPEATS,
        WARMUPS,
    )

    elev_list_avg, elev_list_min, elev_list_max = measure_time(
        lambda: elevation_gain_loss_segments(runner_segments),
        REPEATS,
        WARMUPS,
    )

    elev_numpy_avg, elev_numpy_min, elev_numpy_max = measure_time(
        lambda: elevation_gain_loss_segments_numpy(runner_segments_numpy),
        REPEATS,
        WARMUPS,
    )

    reference_list_avg, reference_list_min, reference_list_max = measure_time(
        lambda: compare_to_reference_route_list(
            reference_segments,
            runner_segments,
        ),
        REFERENCE_REPEATS,
        REFERENCE_WARMUPS,
    )

    reference_numpy_avg, reference_numpy_min, reference_numpy_max = measure_time(
        lambda: compare_to_reference_route_numpy(
            reference_segments,
            runner_segments,
        ),
        REFERENCE_REPEATS,
        REFERENCE_WARMUPS,
    )

    print("BENCHMARK GPX\n")

    print("DYSTANS")
    print(
        f"lista  -> wynik: {distance_list_result_m / 1000.0:.6f} km | "
        f"avg: {dist_list_avg:.6f} ms | min: {dist_list_min:.6f} ms | max: {dist_list_max:.6f} ms"
    )
    print(
        f"numpy  -> wynik: {distance_numpy_result_m / 1000.0:.6f} km | "
        f"avg: {dist_numpy_avg:.6f} ms | min: {dist_numpy_min:.6f} ms | max: {dist_numpy_max:.6f} ms"
    )

    print()
    print("PRZEWYŻSZENIA")
    print(
        f"lista  -> gain: {gain_list:.3f} m | loss: {loss_list:.3f} m | "
        f"avg: {elev_list_avg:.6f} ms | min: {elev_list_min:.6f} ms | max: {elev_list_max:.6f} ms"
    )
    print(
        f"numpy  -> gain: {gain_numpy:.3f} m | loss: {loss_numpy:.3f} m | "
        f"avg: {elev_numpy_avg:.6f} ms | min: {elev_numpy_min:.6f} ms | max: {elev_numpy_max:.6f} ms"
    )

    print()
    print("TRASA REFERENCYJNA")
    print(
        f"lista  -> mean: {reference_list_result['mean_distance_m']:.3f} m | "
        f"max: {reference_list_result['max_distance_m']:.3f} m | "
        f"within_10m: {reference_list_result['within_10m_percent']:.2f}% | "
        f"avg: {reference_list_avg:.6f} ms | "
        f"min: {reference_list_min:.6f} ms | "
        f"max: {reference_list_max:.6f} ms"
    )
    print(
        f"numpy  -> mean: {reference_numpy_result['mean_distance_m']:.3f} m | "
        f"max: {reference_numpy_result['max_distance_m']:.3f} m | "
        f"within_10m: {reference_numpy_result['within_10m_percent']:.2f}% | "
        f"avg: {reference_numpy_avg:.6f} ms | "
        f"min: {reference_numpy_min:.6f} ms | "
        f"max: {reference_numpy_max:.6f} ms"
    )


if __name__ == "__main__":
    main()
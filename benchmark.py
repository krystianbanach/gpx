from time import perf_counter

from gpx_parser import GpxParser, validate_gpx
from gpx_analysis import total_distance_haversine_list, elevation_gain_loss_segments
from gpx_numpy import (
    segments_to_columns,
    total_distance_segments_numpy,
    elevation_gain_loss_segments_numpy,
)

GPX_FILE = 'bieg_10000_1s.gpx'
REPEATS = 10
WARMUPS = 3


def collect_all_points(route_segments):
    points = []
    for segment in route_segments:
        points.extend(segment)
    return points


def load_list_data(gpx_file):
    parser = GpxParser(gpx_file)
    route_segments = parser.load_route_segments()
    all_points = collect_all_points(route_segments)
    validate_gpx(all_points)
    return route_segments


def total_distance_segments_list(route_segments):
    total_km = 0.0
    for segment in route_segments:
        if len(segment) >= 2:
            total_km += total_distance_haversine_list(segment)
    return total_km


def measure_time(func, repeats, warmups):
    for _ in range(warmups):
        func()

    times = []
    for _ in range(repeats):
        start = perf_counter()
        func()
        end = perf_counter()
        times.append((end - start) * 1000)

    avg = sum(times) / len(times)
    return avg, min(times), max(times)


def main():
    route_segments = load_list_data(GPX_FILE)
    segments_numpy = segments_to_columns(route_segments)

    distance_list_result = total_distance_segments_list(route_segments)
    distance_numpy_result = total_distance_segments_numpy(segments_numpy)

    gain_list, loss_list = elevation_gain_loss_segments(route_segments)
    gain_numpy, loss_numpy = elevation_gain_loss_segments_numpy(segments_numpy)

    dist_list_avg, dist_list_min, dist_list_max = measure_time(
        lambda: total_distance_segments_list(route_segments),
        REPEATS,
        WARMUPS,
    )

    dist_numpy_avg, dist_numpy_min, dist_numpy_max = measure_time(
        lambda: total_distance_segments_numpy(segments_numpy),
        REPEATS,
        WARMUPS,
    )

    elev_list_avg, elev_list_min, elev_list_max = measure_time(
        lambda: elevation_gain_loss_segments(route_segments),
        REPEATS,
        WARMUPS,
    )

    elev_numpy_avg, elev_numpy_min, elev_numpy_max = measure_time(
        lambda: elevation_gain_loss_segments_numpy(segments_numpy),
        REPEATS,
        WARMUPS,
    )

    print('BENCHMARK GPX\n')

    print('DYSTANS')
    print(f'lista  -> wynik: {distance_list_result:.6f} km | avg: {dist_list_avg:.6f} ms | min: {dist_list_min:.6f} ms | max: {dist_list_max:.6f} ms')
    print(f'numpy  -> wynik: {distance_numpy_result:.6f} km | avg: {dist_numpy_avg:.6f} ms | min: {dist_numpy_min:.6f} ms | max: {dist_numpy_max:.6f} ms')

    print()
    print('PRZEWYŻSZENIA')
    print(f'lista  -> gain: {gain_list:.3f} m | loss: {loss_list:.3f} m | avg: {elev_list_avg:.6f} ms | min: {elev_list_min:.6f} ms | max: {elev_list_max:.6f} ms')
    print(f'numpy  -> gain: {gain_numpy:.3f} m | loss: {loss_numpy:.3f} m | avg: {elev_numpy_avg:.6f} ms | min: {elev_numpy_min:.6f} ms | max: {elev_numpy_max:.6f} ms')


if __name__ == '__main__':
    main()

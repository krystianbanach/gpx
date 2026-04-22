import os
import sys

from gpx_parser import GpxParser, validate_gpx, load_and_validate_gpx
from gpx_list import compute_route_stats_segments, compute_time_stats_segments, compare_to_reference_route_list
from gpx_report import build_full_report, save_stats_text
from gpx_numpy import compare_to_reference_route_numpy

RUNNER_GPX_FILE = "bieg_10000_1s_zawodnik.gpx"
REFERENCE_GPX_FILE = "bieg_10000_1s.gpx"

def collect_all_points(route_segments):
    points = []
    for segment in route_segments:
        points.extend(segment)
    return points


def main(gpx_file):
    parser = GpxParser(gpx_file)
    route_segments = parser.load_route_segments()
    all_points = collect_all_points(route_segments)
    validate_gpx(all_points)

    route_stats = compute_route_stats_segments(route_segments)
    time_stats = compute_time_stats_segments(route_segments)
    report_text = build_full_report(route_stats, time_stats, route_segments, all_points)

    base_name = os.path.splitext(os.path.basename(gpx_file))[0]
    output_dir = 'wyniki'
    save_stats_text(base_name, output_dir, report_text)
    runner_segments, _ = load_and_validate_gpx(RUNNER_GPX_FILE)
    reference_segments, _ = load_and_validate_gpx(REFERENCE_GPX_FILE)

    reference_list_result = compare_to_reference_route_list(reference_segments, runner_segments)
    reference_numpy_result = compare_to_reference_route_numpy(reference_segments, runner_segments)
    print(report_text)
    print("LIST RESULT")
    print(reference_list_result)
    print("NUMPY RESULT")
    print(reference_numpy_result)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('bieg_10000_1s.gpx')
import os
import sys

from gpx_parser import GpxParser, validate_gpx
from gpx_analysis import compute_route_stats_segments, compute_time_stats_segments
from gpx_report import build_full_report, save_stats_text


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

    print(report_text)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('bieg_10000_1s.gpx')

import datetime
import math
from pathlib import Path


DATA_DIR = Path("data")
POINT_COUNTS = [
    1_000,
    2_000,
    3_000,
    4_000,
    5_000,
    6_000,
    7_000,
    8_000,
    9_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
]

def generate_gpx(filename, num_points, zawodnik=False):
    start_time = datetime.datetime(2026, 3, 12, 18, 26, 20, tzinfo=datetime.timezone.utc)

    center_lat = 52.0
    center_lon = 19.0
    radius = 0.02

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write('<gpx creator="Garmin Connect" version="1.1"\n')
        file.write('     xmlns="http://www.topografix.com/GPX/1/1">\n')
        file.write("  <metadata>\n")
        file.write(f"    <time>{format_time(start_time)}</time>\n")
        file.write("  </metadata>\n")
        file.write("  <trk>\n")
        file.write("    <type>Przykładowy plik GPX</type>\n")
        file.write("    <trkseg>\n")

        for i in range(num_points):
            current_time = start_time + datetime.timedelta(seconds=i)
            angle = i / num_points * 2 * math.pi

            lat = center_lat + math.sin(angle) * radius
            lon = center_lon + math.cos(angle) * radius

            if zawodnik:
                lat += math.sin(i / 80) * 0.00005
                lon += math.cos(i / 90) * 0.00005

            ele = 150 + math.sin(i / 5000) * 20

            file.write(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">\n')
            file.write(f"        <ele>{ele:.1f}</ele>\n")
            file.write(f"        <time>{format_time(current_time)}</time>\n")
            file.write("      </trkpt>\n")

        file.write("    </trkseg>\n")
        file.write("  </trk>\n")
        file.write("</gpx>\n")

    print(f"Utworzono: {filename}")


def format_time(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def generate_dataset():
    for num_points in POINT_COUNTS:
        generate_gpx(DATA_DIR / f"trasa_{num_points}.gpx", num_points)
        generate_gpx(DATA_DIR / f"trasa_{num_points}_zawodnik.gpx", num_points, zawodnik=True)


if __name__ == "__main__":
    generate_dataset()

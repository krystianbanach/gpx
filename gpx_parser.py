from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

MAX_TIME_GAP_S = 300
MAX_DISTANCE_JUMP_M = 100
MAX_ELEVATION_JUMP_M = 4
EARTH_RADIUS_M = 6371008.8


@dataclass(slots=True)
class GpxPoint:
    latitude: float
    longitude: float
    elevation: float | None
    time: datetime | None


def parse_time(time_text):
    if not time_text:
        return None

    time_text = time_text.strip()
    if time_text.endswith("Z"):
        time_text = time_text[:-1] + "+00:00"

    return datetime.fromisoformat(time_text)


class GpxParser:
    def __init__(self, gpx_file):
        self.gpx_file = gpx_file

    def load_route_segments(self):
        tree = ET.parse(self.gpx_file)
        root = tree.getroot()

        ns = {"gpx": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        trkseg_path = ".//gpx:trkseg" if ns else ".//trkseg"
        trkpt_tag = "gpx:trkpt" if ns else "trkpt"
        ele_tag = "gpx:ele" if ns else "ele"
        time_tag = "gpx:time" if ns else "time"

        route_segments = []

        trksegs = root.findall(trkseg_path, ns)
        if not trksegs:
            trksegs = [root]

        for trkseg in trksegs:
            current_segment = []
            prev_time = None

            for trkpt in trkseg.findall(f".//{trkpt_tag}", ns):
                lat = float(trkpt.attrib["lat"])
                lon = float(trkpt.attrib["lon"])

                ele_node = trkpt.find(ele_tag, ns)
                time_node = trkpt.find(time_tag, ns)

                ele = float(ele_node.text) if ele_node is not None and ele_node.text else None
                time_value = parse_time(time_node.text) if time_node is not None and time_node.text else None

                if prev_time is not None and time_value is not None:
                    gap = (time_value - prev_time).total_seconds()
                    if gap < 0 or gap > MAX_TIME_GAP_S:
                        if current_segment:
                            route_segments.append(current_segment)
                        current_segment = []

                current_segment.append(GpxPoint(lat, lon, ele, time_value))
                prev_time = time_value

            if current_segment:
                route_segments.append(current_segment)

        return route_segments


def collect_all_points(route_segments):
    points = []
    for segment in route_segments:
        points.extend(segment)
    return points


def validate_gpx(all_points):
    if not all_points:
        raise ValueError("Plik GPX nie zawiera punktów.")


def load_and_validate_gpx(gpx_file):
    parser = GpxParser(gpx_file)
    route_segments = parser.load_route_segments()
    all_points = collect_all_points(route_segments)
    validate_gpx(all_points)
    return route_segments, all_points
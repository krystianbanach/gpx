import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

MAX_GAP_SECONDS = 60
MAX_ELEVATION_JUMP_M = 4
MAX_DISTANCE_JUMP_M = 100


@dataclass
class GpsPoint:
    latitude: float
    longitude: float
    elevation: float = None
    time: datetime = None


class GpxParser:
    def __init__(self, gpx_file_path):
        self.gpx_file_path = gpx_file_path
        try:
            self.tree = ET.parse(gpx_file_path)
            self.root = self.tree.getroot()
        except Exception as e:
            raise ValueError(f"Nie udało się wczytać pliku GPX: {e}")

    def load_route_segments(self, max_gap_seconds=None):
        if max_gap_seconds is None:
            max_gap_seconds = MAX_GAP_SECONDS

        ns = {'default': 'http://www.topografix.com/GPX/1/1'}
        trksegs = self.root.findall('.//default:trkseg', namespaces=ns)

        route_segments = []
        if trksegs:
            for trkseg in trksegs:
                trkpts = trkseg.findall('default:trkpt', namespaces=ns)
                parsed_points = self._parse_points(trkpts, ns)
                route_segments.extend(split_track_into_segments(parsed_points, max_gap_seconds))
        else:
            trkpts = self.root.findall('.//default:trkpt', namespaces=ns)
            parsed_points = self._parse_points(trkpts, ns)
            route_segments = split_track_into_segments(parsed_points, max_gap_seconds)

        return route_segments

    def _parse_points(self, trkpts, ns):
        parsed_points = []
        for trkpt in trkpts:
            try:
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                ele = trkpt.find('default:ele', namespaces=ns)
                time = trkpt.find('default:time', namespaces=ns)

                parsed_points.append(GpsPoint(
                    latitude=lat,
                    longitude=lon,
                    elevation=float(ele.text) if ele is not None else None,
                    time=datetime.fromisoformat(time.text.replace('Z', '+00:00')) if time is not None else None
                ))
            except Exception as e:
                print(f"Pominięto punkt GPX z błędem: {e}")
        return parsed_points


def split_track_into_segments(track_points, max_gap_seconds=MAX_GAP_SECONDS):
    if not track_points:
        return []

    route_segments = []
    current_segment = [track_points[0]]

    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        if prev_point.time and curr_point.time:
            delta = (curr_point.time - prev_point.time).total_seconds()
            if delta <= 0 or delta > max_gap_seconds:
                route_segments.append(current_segment)
                current_segment = [curr_point]
            else:
                current_segment.append(curr_point)
        else:
            current_segment.append(curr_point)

    if current_segment:
        route_segments.append(current_segment)

    return route_segments


def validate_gpx(points):
    if len(points) < 2:
        raise ValueError('Plik GPX zawiera za mało punktów.')
    if all(pt.time is None for pt in points):
        raise ValueError('Brak danych czasowych w pliku GPX.')

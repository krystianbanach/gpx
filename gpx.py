import xml.etree.ElementTree as ET
import folium
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from geopy.distance import geodesic
from matplotlib import colormaps
from matplotlib import colors as mcolors

@dataclass
class GpsPoint:
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    time: Optional[datetime] = None

    def distance_to(self, other: 'GpsPoint') -> float:
        return geodesic(
            (self.latitude, self.longitude),
            (other.latitude, other.longitude)
        ).km


class GpxParser:
    def __init__(self, gpx_file_path: str):
        self.gpx_file_path = gpx_file_path
        try:
            self.tree = ET.parse(gpx_file_path)
            self.root = self.tree.getroot()
        except Exception as e:
            raise ValueError(f"Nie udało się wczytać pliku GPX: {e}")

    def parse_or_split_segments(self, max_gap_seconds: int = 30) -> List[List[GpsPoint]]:
        ns = {'default': 'http://www.topografix.com/GPX/1/1'}
        trksegs = self.root.findall('.//default:trkseg', namespaces=ns)

        segments = []
        if trksegs:
            for trkseg in trksegs:
                trkpts = trkseg.findall('default:trkpt', namespaces=ns)
                raw_points = self._parse_points(trkpts, ns)
                segments.extend(split_track_into_segments(raw_points, max_gap_seconds))
        else:
            trkpts = self.root.findall('.//default:trkpt', namespaces=ns)
            raw_points = self._parse_points(trkpts, ns)
            segments = split_track_into_segments(raw_points, max_gap_seconds)

        return segments

    def _parse_points(self, trkpts, ns) -> List[GpsPoint]:
        points = []
        for trkpt in trkpts:
            try:
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                ele = trkpt.find('default:ele', namespaces=ns)
                time = trkpt.find('default:time', namespaces=ns)
                points.append(GpsPoint(
                    latitude=lat,
                    longitude=lon,
                    elevation=float(ele.text) if ele is not None else None,
                    time=datetime.fromisoformat(time.text.replace("Z", "+00:00")) if time is not None else None
                ))
            except Exception as e:
                print(f"Pominięto punkt GPX z błędem: {e}")
        return points


def split_track_into_segments(points: List[GpsPoint], max_gap_seconds: int = 30) -> List[List[GpsPoint]]:
    if not points:
        return []

    segments = []
    current_segment = [points[0]]

    for prev, curr in zip(points[:-1], points[1:]):
        if prev.time and curr.time:
            delta = (curr.time - prev.time).total_seconds()
            if delta > max_gap_seconds:
                segments.append(current_segment)
                current_segment = [curr]
            else:
                current_segment.append(curr)
        else:
            current_segment.append(curr)

    if current_segment:
        segments.append(current_segment)

    return segments


def generate_colors(n: int) -> List[str]:
    cmap = colormaps.get_cmap('hsv')
    return [mcolors.to_hex(cmap(i / n)) for i in range(n)]


def plot_route_on_map(segments: List[List[GpsPoint]], map_file: str = "route_map.html"):
    if not segments or not segments[0]:
        print("Brak segmentów do narysowania.")
        return

    start_coords = (segments[0][0].latitude, segments[0][0].longitude)
    m = folium.Map(location=start_coords, zoom_start=13)

    colors = generate_colors(len(segments))

    for idx, segment in enumerate(segments):
        coords = [(p.latitude, p.longitude) for p in segment]
        color = colors[idx]

        folium.PolyLine(coords, color=color, weight=4).add_to(m)
        folium.Marker(coords[0], popup=f"Segment {idx + 1}", icon=folium.Icon(color='blue')).add_to(m)

        if idx == 0:
            folium.Marker(coords[0], popup="Start", icon=folium.Icon(color='green')).add_to(m)
        if idx == len(segments) - 1:
            folium.Marker(coords[-1], popup="Meta", icon=folium.Icon(color='red')).add_to(m)

    m.save(map_file)
    print(f"Mapa zapisana do pliku: {map_file}")


def compute_route_stats(points: List[GpsPoint]) -> dict:
    total_distance = 0.0
    elevation_gain = 0.0
    elevation_loss = 0.0

    for prev, curr in zip(points[:-1], points[1:]):
        total_distance += prev.distance_to(curr)

        if prev.elevation is not None and curr.elevation is not None:
            diff = curr.elevation - prev.elevation
            if diff > 0:
                elevation_gain += diff
            else:
                elevation_loss += abs(diff)

    return {
        "total_distance": total_distance,
        "elevation_gain": elevation_gain,
        "elevation_loss": elevation_loss
    }


def compute_time_stats(points: List[GpsPoint]) -> Optional[Tuple[datetime, datetime, float, str]]:
    times = [p.time for p in points if p.time]
    if len(times) < 2:
        return None

    start_time = times[0]
    end_time = times[-1]
    duration = (end_time - start_time).total_seconds()
    total_distance = sum(points[i - 1].distance_to(points[i]) for i in range(1, len(points)))

    hours = duration / 3600 if duration > 0 else 0
    avg_speed = total_distance / hours if hours > 0 else 0

    if total_distance > 0:
        seconds_per_km = duration / total_distance
        pace = f"{int(seconds_per_km // 60)}:{int(seconds_per_km % 60):02d} min/km"
    else:
        pace = "N/A"

    return (start_time, end_time, avg_speed, pace)


def format_route_stats(stats: dict) -> str:
    return (
        f" Dystans całkowity: {stats['total_distance']:.2f} km\n"
        f" Suma podejść: {stats['elevation_gain']:.0f} m\n"
        f" Suma zejść: {stats['elevation_loss']:.0f} m"
    )


def format_time_stats(stats: Optional[Tuple[datetime, datetime, float, str]]) -> str:
    if not stats:
        return "Brak wystarczających danych czasowych."
    start, end, avg_speed, pace = stats
    return (
        f" Czas rozpoczęcia: {start}\n"
        f" Czas zakończenia: {end}\n"
        f" Czas trwania trasy: {end - start}\n"
        f" Średnia prędkość: {avg_speed:.2f} km/h\n"
        f" Tempo: {pace}"
    )


def main(gpx_file: str, map_file: str):
    parser = GpxParser(gpx_file)
    segments = parser.parse_or_split_segments()
    points = [p for segment in segments for p in segment]

    if not points:
        print("Brak punktów w pliku GPX.")
        return

    #print("\n Przykładowe punkty:")
    #for point in points[:5]:
    #    print(point)

    plot_route_on_map(segments, map_file)
    print(format_route_stats(compute_route_stats(points)))
    print(format_time_stats(compute_time_stats(points)))


if __name__ == "__main__":
    for infile, outfile in [("bieg.gpx", "bieg_map.html"), ("bieg_segmenty.gpx", "bieg_segmenty_map.html")]:
        main(infile, outfile)

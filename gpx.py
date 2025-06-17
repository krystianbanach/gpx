import xml.etree.ElementTree as ET
import folium
from dataclasses import dataclass
from datetime import datetime
from geopy.distance import geodesic
from matplotlib import colormaps
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import os

MAX_GAP_SECONDS = 60
MAX_ELEVATION_JUMP_M = 4     
MAX_DISTANCE_JUMP_M = 100

@dataclass
class GpsPoint:
    latitude: float
    longitude: float
    elevation: float = None
    time: datetime = None

    def distance_to(self, other):
        return geodesic(
            (self.latitude, self.longitude),
            (other.latitude, other.longitude)
        ).km


class GpxParser:
    def __init__(self, gpx_file_path):
        self.gpx_file_path = gpx_file_path
        try:
            self.tree = ET.parse(gpx_file_path)
            self.root = self.tree.getroot()
        except Exception as e:
            raise ValueError(f"Nie udało się wczytać pliku GPX: {e}")

    def parse_or_split_segments(self, max_gap_seconds=None):
        max_gap_seconds = max_gap_seconds if max_gap_seconds is not None else MAX_GAP_SECONDS
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
                    time=datetime.fromisoformat(time.text.replace("Z", "+00:00")) if time is not None else None
                ))
            except Exception as e:
                print(f"Pominięto punkt GPX z błędem: {e}")
        return parsed_points

def split_track_into_segments(track_points, max_gap_seconds=60):
    if not track_points:
        return []

    route_segments = []
    current_segment = [track_points[0]]

    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        if prev_point.time and curr_point.time:
            delta = (curr_point.time - prev_point.time).total_seconds()
            if delta > max_gap_seconds:
                route_segments.append(current_segment)
                current_segment = [curr_point]
            else:
                current_segment.append(curr_point)
        else:
            current_segment.append(curr_point)

    if current_segment:
        route_segments.append(current_segment)

    return route_segments

def compute_route_stats(track_points):
    total_distance = 0.0
    elevation_gain = 0.0
    elevation_loss = 0.0

    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        total_distance += prev_point.distance_to(curr_point)

        if prev_point.elevation is not None and curr_point.elevation is not None:
            diff = curr_point.elevation - prev_point.elevation
            if diff > 0:
                elevation_gain += diff
            else:
                elevation_loss += abs(diff)

    return {
        "total_distance": total_distance,
        "elevation_gain": elevation_gain,
        "elevation_loss": elevation_loss
    }

def compute_time_stats(track_points):
    time_values = [pt.time for pt in track_points if pt.time]
    if len(time_values) < 2:
        return None

    start_time = time_values[0]
    end_time = time_values[-1]
    duration_sec = (end_time - start_time).total_seconds()
    total_distance = sum(track_points[i - 1].distance_to(track_points[i]) for i in range(1, len(track_points)))

    hours = duration_sec / 3600 if duration_sec > 0 else 0
    avg_speed = total_distance / hours if hours > 0 else 0

    if total_distance > 0:
        seconds_per_km = duration_sec / total_distance
        pace = f"{int(seconds_per_km // 60)}:{int(seconds_per_km % 60):02d} min/km"
    else:
        pace = "N/A"

    return (start_time, end_time, avg_speed, pace)

# ==== DETEKCJA ANOMALII ====
def find_position_jumps(track_points):
    return [i for i in range(1, len(track_points))
            if track_points[i - 1].distance_to(track_points[i]) * 1000 > MAX_DISTANCE_JUMP_M]

def count_position_jumps(track_points):
    return len(find_position_jumps(track_points))

def find_elevation_jumps(track_points):
    return [i for i in range(1, len(track_points))
            if track_points[i - 1].elevation is not None and track_points[i].elevation is not None and
               abs(track_points[i].elevation - track_points[i - 1].elevation) > MAX_ELEVATION_JUMP_M]

def count_elevation_jumps(track_points):
    return len(find_elevation_jumps(track_points))

def detect_anomalies(track_points):
    pos_jump_indices = find_position_jumps(track_points)
    elev_jump_indices = find_elevation_jumps(track_points)

    anomalies = {
        "position": pos_jump_indices,
        "elevation": elev_jump_indices
    }
    return anomalies

def get_anomaly_points_with_type(anomalies, all_points):
    labeled = []
    for i in anomalies["position"]:
        labeled.append(("Skok pozycji", all_points[i]))
    for i in anomalies["elevation"]:
        labeled.append(("Skok wysokości", all_points[i]))
    return labeled

def format_route_stats(stats):
    return (
        f" Dystans całkowity: {stats['total_distance']:.2f} km\n"
        f" Suma podejść: {stats['elevation_gain']:.0f} m\n"
        f" Suma zejść: {stats['elevation_loss']:.0f} m"
    )

def format_time_stats(stats):
    if not stats:
        return "Brak wystarczających danych czasowych."
    start, end, avg_speed, pace = stats
    return (
        f" Czas rozpoczęcia: {start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f" Czas zakończenia: {end.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f" Czas trwania trasy: {end - start}\n"
        f" Średnia prędkość: {avg_speed:.2f} km/h\n"
        f" Tempo: {pace}"
    )

def format_gpx_analysis(track_points):
    lines = []
    lines.append("=== ANALIZA DANYCH GPX ===")
    lines.append(f"Liczba punktów: {len(track_points)}")
    lines.append(f"Punkty bez danych o wysokości: {sum(1 for pt in track_points if pt.elevation is None)}")
    lines.append(f"Punkty bez danych o czasie: {sum(1 for pt in track_points if pt.time is None)}")
    lines.append(f"Skoki wysokości > {MAX_ELEVATION_JUMP_M} m: {count_elevation_jumps(track_points)}")
    lines.append(f"Skoki pozycji > {MAX_DISTANCE_JUMP_M} m: {count_position_jumps(track_points)}")
    return "\n".join(lines)

def generate_colors(n):
    cmap = colormaps.get_cmap('hsv')
    return [mcolors.to_hex(cmap(i / n)) for i in range(n)]

def plot_route_on_map(route_segments, map_file="route_map.html", anomaly_points=None):

    if not route_segments or not route_segments[0]:
        print("Brak segmentów do narysowania.")
        return

    start_coords = (route_segments[0][0].latitude, route_segments[0][0].longitude)
    m = folium.Map(location=start_coords, zoom_start=13)

    segment_colors = generate_colors(len(route_segments))

    for idx, route_segment in enumerate(route_segments):
        segment_coords = [(pt.latitude, pt.longitude) for pt in route_segment]
        color = segment_colors[idx]

        folium.PolyLine(segment_coords, color=color, weight=4).add_to(m)
        folium.Marker(segment_coords[0], popup=f"Segment {idx + 1}", icon=folium.Icon(color='blue')).add_to(m)

        if idx == 0:
            folium.Marker(segment_coords[0], popup="Start", icon=folium.Icon(color='green')).add_to(m)
        if idx == len(route_segments) - 1:
            folium.Marker(segment_coords[-1], popup="Meta", icon=folium.Icon(color='red')).add_to(m)
    if anomaly_points:
        for label, pt in anomaly_points:
            color = 'red' if "pozycji" in label else 'orange'
            folium.CircleMarker(
                location=(pt.latitude, pt.longitude),
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=label
            ).add_to(m)
    m.save(map_file)
    print(f"Mapa zapisana do pliku: {map_file}")

def plot_elevation_profile(track_points, file_name="elevation_profile.png", by="distance"):
    x = []
    elevations = []
    total_distance = 0.0

    for i in range(1, len(track_points)):
        prev = track_points[i - 1]
        curr = track_points[i]

        if prev.elevation is not None and curr.elevation is not None:
            if by == "distance":
                total_distance += prev.distance_to(curr)
                x.append(total_distance)
            elif by == "time" and curr.time and prev.time:
                time_diff = (curr.time - track_points[0].time).total_seconds() / 60  
                x.append(time_diff)
            else:
                continue
            elevations.append(curr.elevation)

    if not x or not elevations:
        print("Brak danych wysokościowych do wykresu.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(x, elevations, label="Profil wysokości", linewidth=1.5)
    plt.xlabel("Czas (minuty)" if by == "time" else "Dystans (km)")
    plt.ylabel("Wysokość (m n.p.m.)")
    plt.legend()
    plt.title(f"Profil wysokościowy: Elevation vs {'Czas' if by == 'time' else 'Dystans'}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"Wykres wysokości zapisany jako: {file_name}")


def save_stats_text(base_name, output_dir, route_stats, time_stats, track_points):
    stats_file = os.path.join(output_dir, f"{base_name}_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=== STATYSTYKI TRASY ===\n")
        f.write(format_route_stats(route_stats) + "\n")
        f.write(format_time_stats(time_stats) + "\n\n")
        f.write(format_gpx_analysis(track_points) + "\n")
    print(f"Statystyki zapisane jako: {stats_file}")

def validate_gpx(points):
    if len(points) < 2:
        raise ValueError("Plik GPX zawiera za mało punktów.")
    if all(pt.time is None for pt in points):
        raise ValueError("Brak danych czasowych w pliku GPX.")

def main(gpx_file):
    output_dir = "wyniki"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(gpx_file))[0]
    map_file = os.path.join(output_dir, f"{base_name}_map.html")
    profile_file = os.path.join(output_dir, f"{base_name}_elevation_profile.png")

    parser = GpxParser(gpx_file)
    route_segments = parser.parse_or_split_segments()

    print(f"Liczba segmentów: {len(route_segments)}")

    all_points = [pt for route_segment in route_segments for pt in route_segment]

    validate_gpx(all_points)

    route_stats = compute_route_stats(all_points)
    time_stats = compute_time_stats(all_points)

    print(format_route_stats(route_stats))
    print(format_time_stats(time_stats))

    save_stats_text(base_name, output_dir, route_stats, time_stats, all_points)
    plot_elevation_profile(all_points, os.path.join(output_dir, f"{base_name}_elevation_time.png"), by="time")
    plot_elevation_profile(all_points, os.path.join(output_dir, f"{base_name}_elevation_distance.png"), by="distance")

    anomalies = detect_anomalies(all_points)
    labeled_anomaly_points = get_anomaly_points_with_type(anomalies, all_points)

    plot_route_on_map(route_segments, map_file, labeled_anomaly_points)

if __name__ == "__main__":
    for infile in ["bieg_5km_bs_2.gpx","bieg_5km_zs.gpx"]:
        main(infile)


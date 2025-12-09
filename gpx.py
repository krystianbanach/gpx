import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from geopy.distance import geodesic
import os
import numpy as np
import time

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

def split_track_into_segments(track_points, max_gap_seconds=MAX_GAP_SECONDS):
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
def elevation_gain_loss(track_points):
    elevation_gain = 0.0
    elevation_loss = 0.0

    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        if prev_point.elevation is None or curr_point.elevation is None:
            continue

        diff = curr_point.elevation - prev_point.elevation

        if diff > 0:
            elevation_gain += diff
        else:
            elevation_loss += -diff

    return float(elevation_gain), float(elevation_loss)

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

def compute_route_stats_segments(route_segments):
    total = {"total_distance": 0.0, "elevation_gain": 0.0, "elevation_loss": 0.0}
    for seg in route_segments:
        if not seg or len(seg) < 2:
            continue
        s = compute_route_stats(seg)
        total["total_distance"] += s["total_distance"]
        total["elevation_gain"] += s["elevation_gain"]
        total["elevation_loss"] += s["elevation_loss"]
    return total

def compute_time_stats_segments(route_segments):
    segs = [seg for seg in route_segments if seg and seg[0].time and seg[-1].time]
    if not segs:
        return None

    start_time = segs[0][0].time
    end_time = segs[-1][-1].time
    elapsed_sec = (end_time - start_time).total_seconds()

    moving_sec = sum((seg[-1].time - seg[0].time).total_seconds() for seg in segs)

    dist_km = compute_route_stats_segments(route_segments)["total_distance"]

    def pace(seconds, km):
        if km <= 0 or seconds <= 0:
            return "N/A"
        spk = seconds / km
        return f"{int(spk // 60)}:{int(spk % 60):02d} min/km"

    return {
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_sec": elapsed_sec,
        "moving_sec": moving_sec,
        "stopped_sec": max(0.0, elapsed_sec - moving_sec),
        "distance_km": dist_km,
        "avg_speed_elapsed": dist_km / (elapsed_sec / 3600) if elapsed_sec > 0 else 0.0,
        "avg_speed_moving": dist_km / (moving_sec / 3600) if moving_sec > 0 else 0.0,
        "pace_elapsed": pace(elapsed_sec, dist_km),
        "pace_moving": pace(moving_sec, dist_km),
    }

def format_time_stats_segments(stats):
    if not stats:
        return "Brak wystarczających danych czasowych."
    return (
        f" Czas rozpoczęcia: {stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f" Czas zakończenia: {stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f" Czas (elapsed): {stats['elapsed_sec']:.0f} s\n"
        f" Czas (moving):  {stats['moving_sec']:.0f} s\n"
        f" Postoje/gapy:   {stats['stopped_sec']:.0f} s\n"
        f" Śr. prędkość (elapsed): {stats['avg_speed_elapsed']:.2f} km/h | tempo: {stats['pace_elapsed']}\n"
        f" Śr. prędkość (moving):  {stats['avg_speed_moving']:.2f} km/h | tempo: {stats['pace_moving']}"
    )

def format_segments_summary(route_segments):
    lines = ["SEGMENTY"]
    for idx, seg in enumerate(route_segments, start=1):
        if not seg:
            continue
        rs = compute_route_stats(seg)
        ts = compute_time_stats(seg)
        if ts:
            start, end, avg_speed, pace = ts
            lines.append(
                f" Segment {idx}: punkty={len(seg)} | dystans={rs['total_distance']:.2f} km | czas={end-start} | tempo={pace} | podejścia={rs['elevation_gain']:.1f} m"
            )
        else:
            lines.append(
                f" Segment {idx}: punkty={len(seg)} | dystans={rs['total_distance']:.2f} km | brak czasu"
            )
    return "\n".join(lines)

def elevation_gain_loss_segments(route_segments):
    gain = 0.0
    loss = 0.0
    for seg in route_segments:
        if len(seg) < 2:
            continue
        g, l = elevation_gain_loss(seg)
        gain += g
        loss += l
    return gain, loss

def elevation_gain_loss_numpy_segments(coords_segments):
    gain = 0.0
    loss = 0.0
    for coords in coords_segments:
        if coords.shape[0] < 2:
            continue
        g, l = elevation_gain_loss_numpy(coords)
        gain += g
        loss += l
    return gain, loss

def compare_elevation_gain_loss(route_segments, coords_segments, n=1000):
    g_list, l_list = elevation_gain_loss_segments(route_segments)
    g_np, l_np = elevation_gain_loss_numpy_segments(coords_segments)
    if abs(g_list - g_np) > 1e-6 or abs(l_list - l_np) > 1e-6:
        print(f"UWAGA: LISTA vs NUMPY różnią się! LISTA=({g_list:.3f},{l_list:.3f}) NUMPY=({g_np:.3f},{l_np:.3f})")

    t0 = time.perf_counter()
    for _ in range(n):
        elevation_gain_loss_segments(route_segments)
    t1 = time.perf_counter()
    avg_list_time = (t1 - t0) * 1000.0 / n

    t2 = time.perf_counter()
    for _ in range(n):
        elevation_gain_loss_numpy_segments(coords_segments)
    t3 = time.perf_counter()
    avg_numpy_time = (t3 - t2) * 1000.0 / n

    speedup = avg_list_time / avg_numpy_time if avg_numpy_time > 0 else float("inf")
    return avg_list_time, avg_numpy_time, speedup


#DETEKCJA ANOMALII
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
        f" Suma podejść: {stats['elevation_gain']:.2f} m\n"
        f" Suma zejść: {stats['elevation_loss']:.2f} m"
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
    lines.append("ANALIZA DANYCH GPX")
    lines.append(f"Liczba punktów: {len(track_points)}")
    lines.append(f"Punkty bez danych o wysokości: {sum(1 for pt in track_points if pt.elevation is None)}")
    lines.append(f"Punkty bez danych o czasie: {sum(1 for pt in track_points if pt.time is None)}")
    lines.append(f"Skoki wysokości > {MAX_ELEVATION_JUMP_M} m: {count_elevation_jumps(track_points)}")
    lines.append(f"Skoki pozycji > {MAX_DISTANCE_JUMP_M} m: {count_position_jumps(track_points)}")
    return "\n".join(lines)

def save_stats_text(base_name, output_dir, route_stats, time_stats, track_points):
    stats_file = os.path.join(output_dir, f"{base_name}_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("STATYSTYKI TRASY\n")
        f.write(format_route_stats(route_stats) + "\n")
        f.write(format_time_stats(time_stats) + "\n\n")
        f.write(format_gpx_analysis(track_points) + "\n")
    print(f"Statystyki zapisane jako: {stats_file}")

def validate_gpx(points):
    if len(points) < 2:
        raise ValueError("Plik GPX zawiera za mało punktów.")
    if all(pt.time is None for pt in points):
        raise ValueError("Brak danych czasowych w pliku GPX.")
    
def coords_to_numpy_from_points(points, dtype=np.float64):
    n = len(points)
    X = np.empty((n, 3), dtype=dtype)
    for i, p in enumerate(points):
        X[i, 0] = float(p.latitude)
        X[i, 1] = float(p.longitude)
        X[i, 2] = np.nan if p.elevation is None else float(p.elevation)
    return X

def coords_segments_to_numpy_list(route_segments, dtype=np.float64):
    seg_arrays = []
    for seg in route_segments:
        n = len(seg)
        X = np.empty((n, 3), dtype=dtype)
        for i, p in enumerate(seg):
            X[i, 0] = float(p.latitude)
            X[i, 1] = float(p.longitude)
            X[i, 2] = np.nan if p.elevation is None else float(p.elevation)
        seg_arrays.append(X)
    return seg_arrays

def coords_to_numpy_from_segments(route_segments, dtype=np.float64):
    sep = np.array([[np.nan, np.nan, np.nan]], dtype=dtype)
    parts = []
    for seg in route_segments:
        if not seg:
            continue
        parts.append(coords_to_numpy_from_points(seg, dtype=dtype))
        parts.append(sep)
    if not parts:
        return np.empty((0, 3), dtype=dtype)
    return np.vstack(parts[:-1]) 

def elevation_gain_loss_numpy(coords):
    ele = coords[:, 2] 
    diff = np.diff(ele)

    valid = ~np.isnan(ele)
    diff = diff[valid[:-1] & valid[1:]]  

    elevation_gain = float(diff[diff > 0].sum())
    elevation_loss = float(-diff[diff < 0].sum())
    return elevation_gain, elevation_loss

try:
    from pyproj import Geod
    _GEOD = Geod(ellps="WGS84")
except Exception:
    _GEOD = None

def total_distance_geopy(points):
    return sum(a.distance_to(b) for a, b in zip(points[:-1], points[1:]))

def total_distance_haversine_numpy(coords):
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = np.diff(lat)
    dlon = np.diff(lon)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371.0088  
    return float((R * c).sum())

def total_distance_pyproj(coords):
    if _GEOD is None:
        raise RuntimeError("Brak ")

    lon = coords[:, 1]
    lat = coords[:, 0]

    _, _, dist_m = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return float(dist_m.sum() / 1000.0)

def avg_runtime_ms(fn, repeats):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeats

def compare_distance_methods(route_segments, coords_segments, n=100):

    def geopy_total():
        return sum(
            total_distance_geopy(seg)
            for seg in route_segments
            if len(seg) > 1
        )

    def hav_total():
        return sum(
            total_distance_haversine_numpy(c)
            for c in coords_segments
            if c.shape[0] > 1
        )

    def pyproj_total():
        return sum(
            total_distance_pyproj(c)
            for c in coords_segments
            if c.shape[0] > 1
        )

    d_geopy = geopy_total()
    d_hav = hav_total()

    d_pyproj = None
    if _GEOD is not None:
        d_pyproj = pyproj_total()

    t_geopy = avg_runtime_ms(geopy_total, n)
    t_hav = avg_runtime_ms(hav_total, n)

    t_pyproj = None
    if _GEOD is not None:
        t_pyproj = avg_runtime_ms(pyproj_total, n)

    return {
        "geopy_km": d_geopy,   "geopy_ms": t_geopy,
        "haversine_km": d_hav, "haversine_ms": t_hav,
        "pyproj_km": d_pyproj, "pyproj_ms": t_pyproj,
    }


def format_distance_methods_report(rep):
    base_ms = rep["geopy_ms"]

    def row(name, km, ms):
        if km is None or ms is None:
            return f"{name:<10} brak"
        speedup = (base_ms / ms) if ms > 0 else float("inf")
        return f"{name:<10} dist={km:.5f} km | time={ms:.3f} ms | speedup={speedup:.2f}x"

    lines = []
    lines.append("PORÓWNANIE METOD DYSTANSU (baseline: geopy)")
    lines.append(row("geopy",     rep["geopy_km"],     rep["geopy_ms"]))
    lines.append(row("pyproj",    rep["pyproj_km"],    rep["pyproj_ms"]))
    lines.append(row("haversine", rep["haversine_km"], rep["haversine_ms"]))
    return "\n".join(lines)


def main(gpx_file):
    output_dir = "wyniki"
    os.makedirs(output_dir, exist_ok=True)
    parser = GpxParser(gpx_file)
    route_segments = parser.parse_or_split_segments()
    coords_segments = coords_segments_to_numpy_list(route_segments)
    dist_rep = compare_distance_methods(route_segments, coords_segments, n=100)
    print(format_distance_methods_report(dist_rep),"\n")

    avg_list_time, avg_numpy_time, speedup = compare_elevation_gain_loss(route_segments, coords_segments, n=1000)
    print("Porównanie czasu obliczeń lista/numpy elev_gain_loss")
    print(f"LISTA : {avg_list_time:.4f} ms")
    print(f"NUMPY : {avg_numpy_time:.4f} ms")
    print(f"NumPy jest {speedup:.2f}x szybsze")

if __name__ == "__main__":
    for infile in ["bieg_5km_zs.gpx"]:
        main(infile)

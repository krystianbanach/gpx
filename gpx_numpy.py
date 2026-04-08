import numpy as np


def segment_to_columns(points, dtype=np.float64):
    count = len(points)
    lat = np.empty(count, dtype=dtype)
    lon = np.empty(count, dtype=dtype)
    ele = np.empty(count, dtype=dtype)

    for index, point in enumerate(points):
        lat[index] = float(point.latitude)
        lon[index] = float(point.longitude)
        ele[index] = np.nan if point.elevation is None else float(point.elevation)

    return lat, lon, ele


def segments_to_columns(route_segments, dtype=np.float64):
    segments_numpy = []

    for segment in route_segments:
        if not segment:
            continue
        segments_numpy.append(segment_to_columns(segment, dtype=dtype))

    return segments_numpy


def total_distance_haversine_numpy(lat, lon):
    if lat.shape[0] < 2:
        return 0.0

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0088
    return float((earth_radius_km * c).sum())


def total_distance_segments_numpy(segments_numpy):
    total_km = 0.0

    for lat, lon, ele in segments_numpy:
        if lat.shape[0] < 2:
            continue
        total_km += total_distance_haversine_numpy(lat, lon)

    return total_km


def elevation_gain_loss_numpy(ele):
    if ele.shape[0] < 2:
        return 0.0, 0.0

    diff = np.diff(ele)
    valid = ~np.isnan(ele)
    diff = diff[valid[:-1] & valid[1:]]

    elevation_gain = float(diff[diff > 0].sum())
    elevation_loss = float(-diff[diff < 0].sum())
    return elevation_gain, elevation_loss


def elevation_gain_loss_segments_numpy(segments_numpy):
    gain = 0.0
    loss = 0.0

    for lat, lon, ele in segments_numpy:
        if ele.shape[0] < 2:
            continue
        segment_gain, segment_loss = elevation_gain_loss_numpy(ele)
        gain += segment_gain
        loss += segment_loss

    return gain, loss


def compute_route_stats_segments_numpy(route_segments, dtype=np.float64):
    segments_numpy = segments_to_columns(route_segments, dtype=dtype)
    total_distance = total_distance_segments_numpy(segments_numpy)
    elevation_gain, elevation_loss = elevation_gain_loss_segments_numpy(segments_numpy)

    return {
        'total_distance': float(total_distance),
        'elevation_gain': float(elevation_gain),
        'elevation_loss': float(elevation_loss),
    }

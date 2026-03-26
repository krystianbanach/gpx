import numpy as np


def coords_to_numpy_from_points(points, dtype=np.float64):
    n = len(points)
    coords = np.empty((n, 3), dtype=dtype)
    for i, point in enumerate(points):
        coords[i, 0] = float(point.latitude)
        coords[i, 1] = float(point.longitude)
        coords[i, 2] = np.nan if point.elevation is None else float(point.elevation)
    return coords


def coords_segments_to_numpy_list(route_segments, dtype=np.float64):
    segment_arrays = []
    for segment in route_segments:
        segment_arrays.append(coords_to_numpy_from_points(segment, dtype=dtype))
    return segment_arrays


def coords_to_numpy_from_segments(route_segments, dtype=np.float64):
    separator = np.array([[np.nan, np.nan, np.nan]], dtype=dtype)
    parts = []

    for segment in route_segments:
        if not segment:
            continue
        parts.append(coords_to_numpy_from_points(segment, dtype=dtype))
        parts.append(separator)

    if not parts:
        return np.empty((0, 3), dtype=dtype)

    return np.vstack(parts[:-1])


def elevation_gain_loss_numpy(coords):
    elevation = coords[:, 2]
    diff = np.diff(elevation)
    valid = ~np.isnan(elevation)
    diff = diff[valid[:-1] & valid[1:]]

    elevation_gain = float(diff[diff > 0].sum())
    elevation_loss = float(-diff[diff < 0].sum())
    return elevation_gain, elevation_loss


def elevation_gain_loss_numpy_segments(coords_segments):
    gain = 0.0
    loss = 0.0

    for coords in coords_segments:
        if coords.shape[0] < 2:
            continue
        segment_gain, segment_loss = elevation_gain_loss_numpy(coords)
        gain += segment_gain
        loss += segment_loss

    return gain, loss


def total_distance_haversine_numpy(coords):
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = np.diff(lat)
    dlon = np.diff(lon)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0088
    return float((earth_radius_km * c).sum())


def compute_route_stats_numpy_segments(route_segments, dtype=np.float64):
    total_distance = 0.0
    coords_segments = coords_segments_to_numpy_list(route_segments, dtype=dtype)

    for coords in coords_segments:
        if coords.shape[0] < 2:
            continue
        total_distance += total_distance_haversine_numpy(coords)

    elevation_gain, elevation_loss = elevation_gain_loss_numpy_segments(coords_segments)
    return {
        'total_distance': float(total_distance),
        'elevation_gain': float(elevation_gain),
        'elevation_loss': float(elevation_loss),
    }

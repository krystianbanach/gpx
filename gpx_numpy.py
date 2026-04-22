import numpy as np

from gpx_parser import EARTH_RADIUS_M


def segment_to_columns(segment, dtype=np.float64):
    point_count = len(segment)

    latitudes = np.empty(point_count, dtype=dtype)
    longitudes = np.empty(point_count, dtype=dtype)
    elevations = np.empty(point_count, dtype=dtype)

    for index, point in enumerate(segment):
        latitudes[index] = point.latitude
        longitudes[index] = point.longitude
        elevations[index] = np.nan if point.elevation is None else point.elevation

    return latitudes, longitudes, elevations


def segments_to_columns(route_segments, dtype=np.float64):
    return [segment_to_columns(segment, dtype=dtype) for segment in route_segments if segment]


def total_distance_haversine_numpy(latitudes, longitudes):
    if latitudes.shape[0] < 2:
        return 0.0

    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    delta_lat = np.diff(latitudes_rad)
    delta_lon = np.diff(longitudes_rad)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(latitudes_rad[:-1]) * np.cos(latitudes_rad[1:]) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return float((EARTH_RADIUS_M * c).sum())


def total_distance_segments_numpy(segments_numpy):
    total_distance = 0.0

    for latitudes, longitudes, elevations in segments_numpy:
        total_distance += total_distance_haversine_numpy(latitudes, longitudes)

    return total_distance


def elevation_gain_loss_numpy(elevations):
    if elevations.shape[0] < 2:
        return 0.0, 0.0

    elevation_diff = np.diff(elevations)
    valid_mask = ~np.isnan(elevations)
    elevation_diff = elevation_diff[valid_mask[:-1] & valid_mask[1:]]

    elevation_gain = float(elevation_diff[elevation_diff > 0].sum())
    elevation_loss = float(-elevation_diff[elevation_diff < 0].sum())

    return elevation_gain, elevation_loss


def elevation_gain_loss_segments_numpy(segments_numpy):
    total_gain = 0.0
    total_loss = 0.0

    for latitudes, longitudes, elevations in segments_numpy:
        gain, loss = elevation_gain_loss_numpy(elevations)
        total_gain += gain
        total_loss += loss

    return total_gain, total_loss


def prepare_projection_numpy(origin_lat, origin_lon):
    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    cos_origin_lat = np.cos(origin_lat_rad)
    return origin_lat_rad, origin_lon_rad, cos_origin_lat


def segment_points_to_xy_numpy(segment, origin_lat_rad, origin_lon_rad, cos_origin_lat):
    point_count = len(segment)
    latitudes = np.empty(point_count, dtype=np.float64)
    longitudes = np.empty(point_count, dtype=np.float64)

    for index, point in enumerate(segment):
        latitudes[index] = point.latitude
        longitudes[index] = point.longitude

    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    x = EARTH_RADIUS_M * (longitudes_rad - origin_lon_rad) * cos_origin_lat
    y = EARTH_RADIUS_M * (latitudes_rad - origin_lat_rad)

    return x, y


def build_reference_line_segments_numpy(reference_segments, origin_lat_rad, origin_lon_rad, cos_origin_lat):
    segment_count = 0

    for segment in reference_segments:
        if len(segment) >= 2:
            segment_count += len(segment) - 1

    start_x = np.empty(segment_count, dtype=np.float64)
    start_y = np.empty(segment_count, dtype=np.float64)
    end_x = np.empty(segment_count, dtype=np.float64)
    end_y = np.empty(segment_count, dtype=np.float64)

    offset = 0

    for segment in reference_segments:
        if len(segment) < 2:
            continue

        x, y = segment_points_to_xy_numpy(segment, origin_lat_rad, origin_lon_rad, cos_origin_lat)
        current_count = len(segment) - 1

        start_x[offset:offset + current_count] = x[:-1]
        start_y[offset:offset + current_count] = y[:-1]
        end_x[offset:offset + current_count] = x[1:]
        end_y[offset:offset + current_count] = y[1:]

        offset += current_count

    segment_dx = end_x - start_x
    segment_dy = end_y - start_y
    segment_length_sq = segment_dx * segment_dx + segment_dy * segment_dy

    return start_x, start_y, end_x, end_y, segment_dx, segment_dy, segment_length_sq


def build_runner_xy_numpy(runner_segments, origin_lat_rad, origin_lon_rad, cos_origin_lat):
    point_count = sum(len(segment) for segment in runner_segments)

    runner_x = np.empty(point_count, dtype=np.float64)
    runner_y = np.empty(point_count, dtype=np.float64)

    offset = 0

    for segment in runner_segments:
        if not segment:
            continue

        x, y = segment_points_to_xy_numpy(segment, origin_lat_rad, origin_lon_rad, cos_origin_lat)
        current_count = len(segment)

        runner_x[offset:offset + current_count] = x
        runner_y[offset:offset + current_count] = y

        offset += current_count

    return runner_x, runner_y


def point_to_route_distance_numpy(point_x, point_y, start_x, start_y, end_x, end_y, segment_dx, segment_dy, segment_length_sq):
    point_from_start_x = point_x - start_x
    point_from_start_y = point_y - start_y

    safe_length_sq = np.where(segment_length_sq == 0.0, 1.0, segment_length_sq)
    projection = (point_from_start_x * segment_dx + point_from_start_y * segment_dy) / safe_length_sq
    projection = np.clip(projection, 0.0, 1.0)

    closest_x = start_x + projection * segment_dx
    closest_y = start_y + projection * segment_dy

    zero_mask = segment_length_sq == 0.0
    closest_x = np.where(zero_mask, start_x, closest_x)
    closest_y = np.where(zero_mask, start_y, closest_y)

    dx = point_x - closest_x
    dy = point_y - closest_y
    distance_sq = dx * dx + dy * dy

    return float(np.sqrt(np.min(distance_sq)))


def compare_to_reference_route_numpy(reference_segments, runner_segments, thresholds_m=(5.0, 10.0, 20.0)):
    if not reference_segments:
        raise ValueError("Brak segmentów trasy referencyjnej.")

    if not runner_segments:
        raise ValueError("Brak segmentów trasy zawodnika.")

    first_reference_point = None
    for segment in reference_segments:
        if segment:
            first_reference_point = segment[0]
            break

    if first_reference_point is None:
        raise ValueError("Trasa referencyjna nie zawiera punktów.")

    origin_lat_rad, origin_lon_rad, cos_origin_lat = prepare_projection_numpy(
        first_reference_point.latitude,
        first_reference_point.longitude,
    )

    start_x, start_y, end_x, end_y, segment_dx, segment_dy, segment_length_sq = build_reference_line_segments_numpy(
        reference_segments,
        origin_lat_rad,
        origin_lon_rad,
        cos_origin_lat,
    )

    if start_x.shape[0] == 0:
        raise ValueError("Nie udało się zbudować odcinków trasy referencyjnej.")

    runner_x, runner_y = build_runner_xy_numpy(
        runner_segments,
        origin_lat_rad,
        origin_lon_rad,
        cos_origin_lat,
    )

    if runner_x.shape[0] == 0:
        raise ValueError("Trasa zawodnika nie zawiera punktów do porównania.")

    distances = np.empty(runner_x.shape[0], dtype=np.float64)

    for index in range(runner_x.shape[0]):
        distances[index] = point_to_route_distance_numpy(
            runner_x[index],
            runner_y[index],
            start_x,
            start_y,
            end_x,
            end_y,
            segment_dx,
            segment_dy,
            segment_length_sq,
        )

    result = {
        "mean_distance_m": float(np.mean(distances)),
        "median_distance_m": float(np.median(distances)),
        "max_distance_m": float(np.max(distances)),
    }

    for threshold_m in thresholds_m:
        threshold_label = f"{int(threshold_m)}m"
        result[f"within_{threshold_label}_percent"] = float(np.mean(distances <= threshold_m) * 100.0)

    return result
import math
from statistics import median

from gpx_parser import MAX_DISTANCE_JUMP_M, MAX_ELEVATION_JUMP_M, EARTH_RADIUS_M


def point_distance_haversine_m(point_a, point_b):
    lat1 = math.radians(point_a.latitude)
    lon1 = math.radians(point_a.longitude)
    lat2 = math.radians(point_b.latitude)
    lon2 = math.radians(point_b.longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c


def total_distance_haversine_list(track_points):
    if len(track_points) < 2:
        return 0.0

    total_distance = 0.0
    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        total_distance += point_distance_haversine_m(prev_point, curr_point)

    return total_distance


def compute_route_stats(track_points):
    total_distance = total_distance_haversine_list(track_points)
    elevation_gain = 0.0
    elevation_loss = 0.0

    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        prev_elevation = prev_point.elevation
        curr_elevation = curr_point.elevation

        if prev_elevation is None or curr_elevation is None:
            continue

        elevation_diff = curr_elevation - prev_elevation
        if elevation_diff > 0:
            elevation_gain += elevation_diff
        else:
            elevation_loss += -elevation_diff

    return {
        "total_distance": total_distance,
        "elevation_gain": elevation_gain,
        "elevation_loss": elevation_loss,
    }


def compute_route_stats_segments(route_segments):
    total_distance = 0.0
    elevation_gain = 0.0
    elevation_loss = 0.0

    for segment in route_segments:
        segment_stats = compute_route_stats(segment)
        total_distance += segment_stats["total_distance"]
        elevation_gain += segment_stats["elevation_gain"]
        elevation_loss += segment_stats["elevation_loss"]

    return {
        "total_distance": total_distance,
        "elevation_gain": elevation_gain,
        "elevation_loss": elevation_loss,
    }


def compute_time_stats(track_points):
    time_values = [point.time for point in track_points if point.time is not None]
    if len(time_values) < 2:
        return None

    start_time = time_values[0]
    end_time = time_values[-1]
    duration_sec = (end_time - start_time).total_seconds()

    total_distance_km = total_distance_haversine_list(track_points) / 1000.0
    hours = duration_sec / 3600.0 if duration_sec > 0 else 0.0
    avg_speed = total_distance_km / hours if hours > 0 else 0.0

    if total_distance_km > 0:
        seconds_per_km = duration_sec / total_distance_km
        pace = f"{int(seconds_per_km // 60)}:{int(seconds_per_km % 60):02d} min/km"
    else:
        pace = "N/A"

    return start_time, end_time, avg_speed, pace


def compute_time_stats_segments(route_segments):
    segments_with_time = [
        segment for segment in route_segments
        if segment and segment[0].time is not None and segment[-1].time is not None
    ]
    if not segments_with_time:
        return None

    start_time = segments_with_time[0][0].time
    end_time = segments_with_time[-1][-1].time
    elapsed_sec = (end_time - start_time).total_seconds()

    moving_sec = 0.0
    for segment in segments_with_time:
        moving_sec += (segment[-1].time - segment[0].time).total_seconds()

    distance_km = compute_route_stats_segments(route_segments)["total_distance"] / 1000.0

    def format_pace(seconds, km):
        if km <= 0 or seconds <= 0:
            return "N/A"
        seconds_per_km = seconds / km
        return f"{int(seconds_per_km // 60)}:{int(seconds_per_km % 60):02d} min/km"

    return {
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_sec": elapsed_sec,
        "moving_sec": moving_sec,
        "stopped_sec": max(0.0, elapsed_sec - moving_sec),
        "distance_km": distance_km,
        "avg_speed_elapsed": distance_km / (elapsed_sec / 3600.0) if elapsed_sec > 0 else 0.0,
        "avg_speed_moving": distance_km / (moving_sec / 3600.0) if moving_sec > 0 else 0.0,
        "pace_elapsed": format_pace(elapsed_sec, distance_km),
        "pace_moving": format_pace(moving_sec, distance_km),
    }


def elevation_gain_loss_segments(route_segments):
    total_gain = 0.0
    total_loss = 0.0

    for segment in route_segments:
        for prev_point, curr_point in zip(segment[:-1], segment[1:]):
            prev_elevation = prev_point.elevation
            curr_elevation = curr_point.elevation

            if prev_elevation is None or curr_elevation is None:
                continue

            elevation_diff = curr_elevation - prev_elevation
            if elevation_diff > 0:
                total_gain += elevation_diff
            else:
                total_loss += -elevation_diff

    return total_gain, total_loss


def find_position_jumps(track_points):
    result = []

    for index in range(1, len(track_points)):
        if point_distance_haversine_m(track_points[index - 1], track_points[index]) > MAX_DISTANCE_JUMP_M:
            result.append(index)

    return result


def count_position_jumps(track_points):
    return len(find_position_jumps(track_points))


def find_elevation_jumps(track_points):
    result = []

    for index in range(1, len(track_points)):
        prev_elevation = track_points[index - 1].elevation
        curr_elevation = track_points[index].elevation

        if prev_elevation is None or curr_elevation is None:
            continue

        if abs(curr_elevation - prev_elevation) > MAX_ELEVATION_JUMP_M:
            result.append(index)

    return result


def count_elevation_jumps(track_points):
    return len(find_elevation_jumps(track_points))


def prepare_projection(origin_lat, origin_lon):
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)
    cos_origin_lat = math.cos(origin_lat_rad)
    return origin_lat_rad, origin_lon_rad, cos_origin_lat


def point_to_xy_m(point, origin_lat_rad, origin_lon_rad, cos_origin_lat):
    lat_rad = math.radians(point.latitude)
    lon_rad = math.radians(point.longitude)

    x = EARTH_RADIUS_M * (lon_rad - origin_lon_rad) * cos_origin_lat
    y = EARTH_RADIUS_M * (lat_rad - origin_lat_rad)

    return x, y


def build_reference_line_segments(reference_segments, origin_lat_rad, origin_lon_rad, cos_origin_lat):
    line_segments = []

    for segment in reference_segments:
        if len(segment) < 2:
            continue

        for prev_point, curr_point in zip(segment[:-1], segment[1:]):
            start_x, start_y = point_to_xy_m(prev_point, origin_lat_rad, origin_lon_rad, cos_origin_lat)
            end_x, end_y = point_to_xy_m(curr_point, origin_lat_rad, origin_lon_rad, cos_origin_lat)

            segment_dx = end_x - start_x
            segment_dy = end_y - start_y
            segment_length_sq = segment_dx * segment_dx + segment_dy * segment_dy

            line_segments.append(
                (start_x, start_y, end_x, end_y, segment_dx, segment_dy, segment_length_sq)
            )

    return line_segments


def point_to_route_distance_m(point_x, point_y, reference_line_segments):
    best_distance_sq = float("inf")

    for start_x, start_y, end_x, end_y, segment_dx, segment_dy, segment_length_sq in reference_line_segments:
        point_from_start_x = point_x - start_x
        point_from_start_y = point_y - start_y

        if segment_length_sq == 0.0:
            dx = point_x - start_x
            dy = point_y - start_y
            distance_sq = dx * dx + dy * dy
        else:
            projection = (
                point_from_start_x * segment_dx + point_from_start_y * segment_dy
            ) / segment_length_sq

            if projection < 0.0:
                closest_x = start_x
                closest_y = start_y
            elif projection > 1.0:
                closest_x = end_x
                closest_y = end_y
            else:
                closest_x = start_x + projection * segment_dx
                closest_y = start_y + projection * segment_dy

            dx = point_x - closest_x
            dy = point_y - closest_y
            distance_sq = dx * dx + dy * dy

        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq

    return math.sqrt(best_distance_sq)


def compare_to_reference_route_list(
    reference_segments,
    runner_segments,
    thresholds_m=(5.0, 10.0, 20.0),
):
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

    origin_lat_rad, origin_lon_rad, cos_origin_lat = prepare_projection(
        first_reference_point.latitude,
        first_reference_point.longitude,
    )

    reference_line_segments = build_reference_line_segments(
        reference_segments,
        origin_lat_rad,
        origin_lon_rad,
        cos_origin_lat,
    )

    if not reference_line_segments:
        raise ValueError("Nie udało się zbudować odcinków trasy referencyjnej.")

    distances = []
    threshold_counts = [0] * len(thresholds_m)

    for segment in runner_segments:
        for point in segment:
            point_x, point_y = point_to_xy_m(point, origin_lat_rad, origin_lon_rad, cos_origin_lat)
            distance = point_to_route_distance_m(point_x, point_y, reference_line_segments)
            distances.append(distance)

            for index, threshold_m in enumerate(thresholds_m):
                if distance <= threshold_m:
                    threshold_counts[index] += 1

    if not distances:
        raise ValueError("Trasa zawodnika nie zawiera punktów do porównania.")

    point_count = len(distances)

    result = {
        "mean_distance_m": sum(distances) / point_count,
        "median_distance_m": median(distances),
        "max_distance_m": max(distances),
    }

    for index, threshold_m in enumerate(thresholds_m):
        threshold_label = f"{int(threshold_m)}m"
        result[f"within_{threshold_label}_percent"] = (threshold_counts[index] / point_count) * 100.0

    return result
import math

from gpx_parser import MAX_DISTANCE_JUMP_M, MAX_ELEVATION_JUMP_M

EARTH_RADIUS_KM = 6371.0088


def point_distance_haversine_km(point_a, point_b):
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
    return EARTH_RADIUS_KM * c


def total_distance_haversine_list(track_points):
    if len(track_points) < 2:
        return 0.0

    total_distance = 0.0
    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        total_distance += point_distance_haversine_km(prev_point, curr_point)
    return total_distance


def compute_route_stats(track_points):
    total_distance = total_distance_haversine_list(track_points)
    elevation_gain = 0.0
    elevation_loss = 0.0

    for prev_point, curr_point in zip(track_points[:-1], track_points[1:]):
        if prev_point.elevation is not None and curr_point.elevation is not None:
            diff = curr_point.elevation - prev_point.elevation
            if diff > 0:
                elevation_gain += diff
            else:
                elevation_loss += abs(diff)

    return {
        'total_distance': total_distance,
        'elevation_gain': elevation_gain,
        'elevation_loss': elevation_loss,
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
    total_distance = total_distance_haversine_list(track_points)

    hours = duration_sec / 3600 if duration_sec > 0 else 0
    avg_speed = total_distance / hours if hours > 0 else 0

    if total_distance > 0:
        seconds_per_km = duration_sec / total_distance
        pace = f"{int(seconds_per_km // 60)}:{int(seconds_per_km % 60):02d} min/km"
    else:
        pace = 'N/A'

    return start_time, end_time, avg_speed, pace


def compute_route_stats_segments(route_segments):
    total = {'total_distance': 0.0, 'elevation_gain': 0.0, 'elevation_loss': 0.0}
    for segment in route_segments:
        if not segment or len(segment) < 2:
            continue
        stats = compute_route_stats(segment)
        total['total_distance'] += stats['total_distance']
        total['elevation_gain'] += stats['elevation_gain']
        total['elevation_loss'] += stats['elevation_loss']
    return total


def compute_time_stats_segments(route_segments):
    segments_with_time = [segment for segment in route_segments if segment and segment[0].time and segment[-1].time]
    if not segments_with_time:
        return None

    start_time = segments_with_time[0][0].time
    end_time = segments_with_time[-1][-1].time
    elapsed_sec = (end_time - start_time).total_seconds()
    moving_sec = sum((segment[-1].time - segment[0].time).total_seconds() for segment in segments_with_time)
    distance_km = compute_route_stats_segments(route_segments)['total_distance']

    def pace(seconds, km):
        if km <= 0 or seconds <= 0:
            return 'N/A'
        seconds_per_km = seconds / km
        return f"{int(seconds_per_km // 60)}:{int(seconds_per_km % 60):02d} min/km"

    return {
        'start_time': start_time,
        'end_time': end_time,
        'elapsed_sec': elapsed_sec,
        'moving_sec': moving_sec,
        'stopped_sec': max(0.0, elapsed_sec - moving_sec),
        'distance_km': distance_km,
        'avg_speed_elapsed': distance_km / (elapsed_sec / 3600) if elapsed_sec > 0 else 0.0,
        'avg_speed_moving': distance_km / (moving_sec / 3600) if moving_sec > 0 else 0.0,
        'pace_elapsed': pace(elapsed_sec, distance_km),
        'pace_moving': pace(moving_sec, distance_km),
    }


def elevation_gain_loss_segments(route_segments):
    gain = 0.0
    loss = 0.0
    for segment in route_segments:
        if len(segment) < 2:
            continue
        segment_gain, segment_loss = elevation_gain_loss(segment)
        gain += segment_gain
        loss += segment_loss
    return gain, loss


def find_position_jumps(track_points):
    return [
        i for i in range(1, len(track_points))
        if point_distance_haversine_km(track_points[i - 1], track_points[i]) * 1000 > MAX_DISTANCE_JUMP_M
    ]


def count_position_jumps(track_points):
    return len(find_position_jumps(track_points))


def find_elevation_jumps(track_points):
    return [
        i for i in range(1, len(track_points))
        if track_points[i - 1].elevation is not None
        and track_points[i].elevation is not None
        and abs(track_points[i].elevation - track_points[i - 1].elevation) > MAX_ELEVATION_JUMP_M
    ]


def count_elevation_jumps(track_points):
    return len(find_elevation_jumps(track_points))


def detect_anomalies(track_points):
    return {
        'position': find_position_jumps(track_points),
        'elevation': find_elevation_jumps(track_points),
    }


def get_anomaly_points_with_type(anomalies, all_points):
    labeled = []
    for index in anomalies['position']:
        labeled.append(('Skok pozycji', all_points[index]))
    for index in anomalies['elevation']:
        labeled.append(('Skok wysokości', all_points[index]))
    return labeled

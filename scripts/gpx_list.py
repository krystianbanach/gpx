import math
from pyproj import Transformer


R = 6371000.0


TRANSFORMER_2180 = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:2180",
    always_xy=True,
)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Oblicza odległość między dwoma punktami GPS wzorem Haversine'a.
    Zwraca wynik w metrach.
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def haversine_total_distance(latitudes, longitudes):
    """
    Oblicza całkowity dystans trasy jako sumę odległości między kolejnymi punktami GPS.
    """
    total_distance = 0.0

    for i in range(1, len(latitudes)):
        distance = haversine_distance(
            latitudes[i - 1],
            longitudes[i - 1],
            latitudes[i],
            longitudes[i],
        )

        total_distance += distance

    return total_distance


def elevation_gain_loss(elevations):
    """
    Oblicza sumę podejść i zejść na podstawie różnic wysokości między kolejnymi punktami.
    Zwraca elevation_gain i elevation_loss w metrach.
    """
    elevation_gain = 0.0
    elevation_loss = 0.0

    for i in range(1, len(elevations)):
        difference = elevations[i] - elevations[i - 1]

        if difference > 0:
            elevation_gain += difference
        elif difference < 0:
            elevation_loss += abs(difference)

    return elevation_gain, elevation_loss


def route_to_euclidean(latitudes, longitudes):
    """
    Przekształca współrzędne GPS z WGS84 do układu metrycznego EPSG:2180.
    Zwraca dwie listy: xs i ys.
    """
    xs = []
    ys = []

    for lat, lon in zip(latitudes, longitudes):
        x, y = TRANSFORMER_2180.transform(lon, lat)
        xs.append(x)
        ys.append(y)

    return xs, ys



def build_segments(xs, ys):
    """
    Buduje segmenty trasy referencyjnej z kolejnych punktów w układzie metrycznym.
    """
    segments = []

    for i in range(1, len(xs)):
        ax = xs[i - 1]
        ay = ys[i - 1]

        bx = xs[i]
        by = ys[i]

        abx = bx - ax
        aby = by - ay
        ab_len2 = abx * abx + aby * aby

        segments.append((ax, ay, bx, by, abx, aby, ab_len2))

    return segments


def distance_to_segments(px, py, segments):
    """
    Oblicza najmniejszą odległość punktu od segmentów trasy referencyjnej.
    """
    min_dist2 = None

    for ax, ay, bx, by, abx, aby, ab_len2 in segments:
        apx = px - ax
        apy = py - ay

        if ab_len2 == 0.0:
            dx = px - ax
            dy = py - ay
        else:
            t = (apx * abx + apy * aby) / ab_len2

            if t < 0.0:
                dx = px - ax
                dy = py - ay
            elif t > 1.0:
                dx = px - bx
                dy = py - by
            else:
                closest_x = ax + t * abx
                closest_y = ay + t * aby

                dx = px - closest_x
                dy = py - closest_y

        dist2 = dx * dx + dy * dy

        if min_dist2 is None or dist2 < min_dist2:
            min_dist2 = dist2

    return math.sqrt(min_dist2)


def compare_routes(
    reference_latitudes,
    reference_longitudes,
    runner_latitudes,
    runner_longitudes,
    thresholds_m=(5.0, 10.0, 20.0),
):
    reference_xs, reference_ys = route_to_euclidean(
        reference_latitudes,
        reference_longitudes,
    )

    runner_xs, runner_ys = route_to_euclidean(
        runner_latitudes,
        runner_longitudes,
    )

    segments = build_segments(reference_xs, reference_ys)

    total_distance_from_route = 0.0
    max_distance = 0.0

    threshold_counts = {}

    for threshold in thresholds_m:
        threshold_counts[threshold] = 0

    for i in range(len(runner_xs)):
        distance = distance_to_segments(
            runner_xs[i],
            runner_ys[i],
            segments,
        )

        total_distance_from_route += distance

        if distance > max_distance:
            max_distance = distance

        for threshold in thresholds_m:
            if distance <= threshold:
                threshold_counts[threshold] += 1

    points_count = len(runner_xs)

    result = {
        "mean_distance_m": total_distance_from_route / points_count,
        "max_distance_m": max_distance,
    }

    for threshold in thresholds_m:
        key = f"within_{int(threshold)}m_percent"
        result[key] = threshold_counts[threshold] / points_count * 100.0

    return result


def frechet_distance(
    reference_latitudes,
    reference_longitudes,
    runner_latitudes,
    runner_longitudes,
):
    reference_xs, reference_ys = route_to_euclidean(
        reference_latitudes,
        reference_longitudes,
    )

    runner_xs, runner_ys = route_to_euclidean(
        runner_latitudes,
        runner_longitudes,
    )

    n = len(reference_xs)
    m = len(runner_xs)

    ca = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            dx = reference_xs[i] - runner_xs[j]
            dy = reference_ys[i] - runner_ys[j]

            dist2 = dx * dx + dy * dy

            if i == 0 and j == 0:
                ca[i][j] = dist2
            elif i == 0:
                ca[i][j] = max(ca[i][j - 1], dist2)
            elif j == 0:
                ca[i][j] = max(ca[i - 1][j], dist2)
            else:
                ca[i][j] = max(
                    min(
                        ca[i - 1][j],
                        ca[i - 1][j - 1],
                        ca[i][j - 1],
                    ),
                    dist2,
                )

    return math.sqrt(ca[n - 1][m - 1])

def segment_durations(
    latitudes,
    longitudes,
    times,
    boundaries,
):
    durations = []

    cumulative_distance = 0.0
    boundary_index = 1
    segment_start_time = times[0]

    for i in range(1, len(latitudes)):
        previous_distance = cumulative_distance

        distance = haversine_distance(
            latitudes[i - 1],
            longitudes[i - 1],
            latitudes[i],
            longitudes[i],
        )

        cumulative_distance += distance

        while (
            boundary_index < len(boundaries)
            and boundaries[boundary_index] <= cumulative_distance
        ):
            target_distance = boundaries[boundary_index]

            ratio = (
                (target_distance - previous_distance)
                / distance
            )

            boundary_time = (
                times[i - 1]
                + ratio * (times[i] - times[i - 1])
            )

            durations.append(boundary_time - segment_start_time)

            segment_start_time = boundary_time
            boundary_index += 1

    return durations


def compare_segment_speeds(
    reference_latitudes,
    reference_longitudes,
    reference_times,
    runner_latitudes,
    runner_longitudes,
    runner_times,
    segment_length_m=500.0,
):
    reference_distance = haversine_total_distance(
        reference_latitudes,
        reference_longitudes,
    )

    runner_distance = haversine_total_distance(
        runner_latitudes,
        runner_longitudes,
    )

    common_distance = min(reference_distance, runner_distance)

    boundaries = [0.0]
    next_boundary = segment_length_m

    while next_boundary < common_distance:
        boundaries.append(next_boundary)
        next_boundary += segment_length_m

    boundaries.append(common_distance)

    reference_durations = segment_durations(
        reference_latitudes,
        reference_longitudes,
        reference_times,
        boundaries,
    )

    runner_durations = segment_durations(
        runner_latitudes,
        runner_longitudes,
        runner_times,
        boundaries,
    )

    segments_count = min(len(reference_durations), len(runner_durations))

    total_weighted_speed_ratio = 0.0
    total_weighted_speed_ratio_square = 0.0
    total_distance = 0.0

    max_speed_ratio = 0.0
    min_speed_ratio = 0.0

    for i in range(segments_count):
        segment_distance = boundaries[i + 1] - boundaries[i]
        speed_ratio = reference_durations[i] / runner_durations[i]

        total_weighted_speed_ratio += speed_ratio * segment_distance
        total_weighted_speed_ratio_square += (
            speed_ratio * speed_ratio * segment_distance
        )
        total_distance += segment_distance

        if i == 0 or speed_ratio > max_speed_ratio:
            max_speed_ratio = speed_ratio

        if i == 0 or speed_ratio < min_speed_ratio:
            min_speed_ratio = speed_ratio

    mean_speed_ratio = total_weighted_speed_ratio / total_distance

    variance_speed_ratio = (
        total_weighted_speed_ratio_square / total_distance
        - mean_speed_ratio * mean_speed_ratio
    )

    std_speed_ratio = variance_speed_ratio ** 0.5

    return (
        segments_count,
        mean_speed_ratio,
        max_speed_ratio,
        min_speed_ratio,
        std_speed_ratio,
    )
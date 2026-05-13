import math
from pyproj import Transformer


R = 6371000


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
    Zwraca współrzędne x, y w metrach.
    """
    points_xy = []

    for lat, lon in zip(latitudes, longitudes):
        x, y = TRANSFORMER_2180.transform(lon, lat)
        points_xy.append((x, y))

    return points_xy



def build_segments(points_xy):
    """
    Buduje segmenty trasy referencyjnej z kolejnych punktów w układzie metrycznym.
    Zwraca opis odcinków potrzebny do obliczania odległości punkt-odcinek.
    """
    segments = []
    for i in range(1, len(points_xy)):
        ax, ay = points_xy[i - 1]
        bx, by = points_xy[i]

        abx = bx - ax
        aby = by - ay
        ab_len2 = abx * abx + aby * aby

        segments.append((ax, ay, bx, by, abx, aby, ab_len2))

    return segments


def distance_to_segments(px, py, segments):
    """
    Oblicza najmniejszą odległość punktu od segmentów trasy referencyjnej.
    Zwraca najkrótszą odległość w metrach.
    """
    min_distance_squared = None

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

        distance_squared = dx * dx + dy * dy

        if min_distance_squared is None or distance_squared < min_distance_squared:
            min_distance_squared = distance_squared

    return math.sqrt(min_distance_squared)


def compare_routes(
    reference_lat,
    reference_lon,
    runner_lat,
    runner_lon,
    thresholds_m=(5.0, 10.0, 20.0),
):
    """
    Porównuje trasę zawodnika z trasą referencyjną na podstawie odległości punktów od segmentów.
    Zwraca średnią odległość, maksymalną odległość oraz procent punktów w zadanych progach.
    """
    reference_xy = route_to_euclidean(
        reference_lat,
        reference_lon,
    )

    runner_xy = route_to_euclidean(
        runner_lat,
        runner_lon,
    )

    segments = build_segments(reference_xy)

    total_distance_from_route = 0.0
    max_distance = 0.0

    threshold_counts = {}

    for threshold in thresholds_m:
        threshold_counts[threshold] = 0

    for px, py in runner_xy:
        distance = distance_to_segments(px, py, segments)

        total_distance_from_route += distance

        if distance > max_distance:
            max_distance = distance

        for threshold in thresholds_m:
            if distance <= threshold:
                threshold_counts[threshold] += 1

    points_count = len(runner_xy)

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
    """
    Liczy dyskretną odległość Frécheta dla punktów w układzie metrycznym.
    """

    reference_xy = route_to_euclidean(
        reference_latitudes,
        reference_longitudes,
    )

    runner_xy = route_to_euclidean(
        runner_latitudes,
        runner_longitudes,
    )

    n = len(reference_xy)
    m = len(runner_xy)

    ca = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            ax, ay = reference_xy[i]
            bx, by = runner_xy[j]

            dx = ax - bx
            dy = ay - by

            dist2 = dx * dx + dy * dy

            if i == 0 and j == 0:
                ca[i][j] = dist2

            elif i == 0:
                ca[i][j] = max(
                    ca[i][j - 1],
                    dist2,
                )

            elif j == 0:
                ca[i][j] = max(
                    ca[i - 1][j],
                    dist2,
                )

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
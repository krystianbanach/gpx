import math
import numpy as np
from pyproj import Transformer


R = 6371000.0


TRANSFORMER_2180 = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:2180",
    always_xy=True,
)


def haversine_distance_np(lat1, lon1, lat2, lon2):
    """
    Oblicza odległość między punktami GPS wzorem Haversine'a.
    Zwraca wynik w metrach.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


def haversine_total_distance_np(latitudes, longitudes):
    """
    Oblicza całkowity dystans trasy jako sumę odległości między kolejnymi punktami GPS.
    """
    if len(latitudes) < 2:
        return 0.0

    distances = haversine_distance_np(
        latitudes[:-1],
        longitudes[:-1],
        latitudes[1:],
        longitudes[1:],
    )

    return float(np.sum(distances))


def elevation_gain_loss_np(elevations):
    """
    Oblicza sumę podejść i zejść na podstawie różnic wysokości między kolejnymi punktami.
    Zwraca elevation_gain i elevation_loss w metrach.
    """
    if len(elevations) < 2:
        return 0.0, 0.0

    differences = elevations[1:] - elevations[:-1]

    elevation_gain = np.sum(differences[differences > 0.0])
    elevation_loss = np.sum(np.abs(differences[differences < 0.0]))

    return float(elevation_gain), float(elevation_loss)


def route_to_euclidean_np(latitudes, longitudes):
    """
    Przekształca tablice GPS do układu EPSG:2180.
    Zwraca tablice xs i ys w metrach.
    """
    xs, ys = TRANSFORMER_2180.transform(longitudes, latitudes)

    return xs, ys


def build_segments_np(xs, ys):
    """
    Buduje segmenty trasy referencyjnej z kolejnych punktów w układzie metrycznym.
    Zwraca opis odcinków potrzebny do obliczania odległości punkt-odcinek.
    """
    ax = xs[:-1]
    ay = ys[:-1]

    bx = xs[1:]
    by = ys[1:]

    abx = bx - ax
    aby = by - ay

    ab_len2 = abx * abx + aby * aby

    return ax, ay, bx, by, abx, aby, ab_len2


def distance_to_segments_np(px, py, segments):
    """
    Oblicza najmniejszą odległość punktu od segmentów trasy referencyjnej.
    Zwraca najkrótszą odległość w metrach.
    """
    ax, ay, bx, by, abx, aby, ab_len2 = segments

    apx = px - ax
    apy = py - ay

    t = np.zeros_like(ab_len2)

    valid = ab_len2 != 0.0

    t[valid] = (
        apx[valid] * abx[valid]
        + apy[valid] * aby[valid]
    ) / ab_len2[valid]

    t = np.clip(t, 0.0, 1.0)

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    dx = px - closest_x
    dy = py - closest_y

    dist2 = dx * dx + dy * dy

    min_dist2 = np.min(dist2)

    return float(np.sqrt(min_dist2))


def compare_routes_np(
    reference_latitudes,
    reference_longitudes,
    runner_latitudes,
    runner_longitudes,
    thresholds_m=(5.0, 10.0, 20.0),
):
    """
    Porównuje trasę zawodnika z trasą referencyjną na podstawie odległości punktów od segmentów.
    Zwraca średnią odległość, maksymalną odległość oraz procent punktów w zadanych progach.
    """
    reference_xs, reference_ys = route_to_euclidean_np(
        reference_latitudes,
        reference_longitudes,
    )

    runner_xs, runner_ys = route_to_euclidean_np(
        runner_latitudes,
        runner_longitudes,
    )

    segments = build_segments_np(reference_xs, reference_ys)

    distances = np.empty(len(runner_xs), dtype=np.float64)

    for i in range(len(runner_xs)):
        distances[i] = distance_to_segments_np(
            runner_xs[i],
            runner_ys[i],
            segments,
        )

    result = {
        "mean_distance_m": float(np.mean(distances)),
        "max_distance_m": float(np.max(distances)),
    }

    for threshold in thresholds_m:
        key = f"within_{int(threshold)}m_percent"
        result[key] = float(
            np.sum(distances <= threshold) / len(distances) * 100.0
        )

    return result

def frechet_distance_np(
    reference_latitudes,
    reference_longitudes,
    runner_latitudes,
    runner_longitudes,
):
    """
    Liczy dyskretną odległość Frécheta dla punktów w układzie metrycznym.
    """

    reference_xs, reference_ys = route_to_euclidean_np(
        reference_latitudes,
        reference_longitudes,
    )

    runner_xs, runner_ys = route_to_euclidean_np(
        runner_latitudes,
        runner_longitudes,
    )

    n = len(reference_xs)
    m = len(runner_xs)

    ca = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            dx = reference_xs[i] - runner_xs[j]
            dy = reference_ys[i] - runner_ys[j]

            dist2 = dx * dx + dy * dy

            if i == 0 and j == 0:
                ca[i, j] = dist2

            elif i == 0:
                ca[i, j] = max(
                    ca[i, j - 1],
                    dist2,
                )

            elif j == 0:
                ca[i, j] = max(
                    ca[i - 1, j],
                    dist2,
                )

            else:
                ca[i, j] = max(
                    min(
                        ca[i - 1, j],
                        ca[i - 1, j - 1],
                        ca[i, j - 1],
                    ),
                    dist2,
                )

    return float(np.sqrt(ca[n - 1, m - 1]))
import math
import numpy as np
from numba import njit
from pyproj import Transformer


R = 6371000.0


TRANSFORMER_2180 = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:2180",
    always_xy=True,
)

@njit
def haversine_distance_numba(lat1, lon1, lat2, lon2):
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
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2.0) ** 2
    )

    if a < 0.0:
        a = 0.0
    elif a > 1.0:
        a = 1.0

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c


@njit
def haversine_total_distance_numba(latitudes, longitudes):
    """
    Oblicza całkowity dystans trasy jako sumę odległości między kolejnymi punktami GPS.
    """
    total_distance = 0.0

    for i in range(1, len(latitudes)):
        distance = haversine_distance_numba(
            latitudes[i - 1],
            longitudes[i - 1],
            latitudes[i],
            longitudes[i],
        )

        total_distance += distance

    return total_distance


@njit
def elevation_gain_loss_numba(elevations):
    """
    Oblicza sumę podejść i zejść na podstawie różnic wysokości między kolejnymi punktami.
    Zwraca elevation_gain i elevation_loss w metrach.
    """
    elevation_gain = 0.0
    elevation_loss = 0.0

    for i in range(1, len(elevations)):
        difference = elevations[i] - elevations[i - 1]

        if difference > 0.0:
            elevation_gain += difference
        elif difference < 0.0:
            elevation_loss += abs(difference)

    return elevation_gain, elevation_loss


def route_to_euclidean_numba(latitudes, longitudes):
    """
    Przekształca współrzędne GPS z WGS84 do układu metrycznego EPSG:2180.
    Zwraca współrzędne x, y w metrach.
    """
    xs, ys = TRANSFORMER_2180.transform(longitudes, latitudes)

    return xs, ys


@njit
def build_segments_numba(xs, ys):
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


@njit
def distance_to_segments_numba(px, py, ax, ay, bx, by, abx, aby, ab_len2):
    """
    Oblicza najmniejszą odległość punktu od segmentów trasy referencyjnej.
    Zwraca najkrótszą odległość w metrach.
    """
    min_dist2 = -1.0

    for i in range(len(ax)):
        apx = px - ax[i]
        apy = py - ay[i]

        if ab_len2[i] == 0.0:
            dx = px - ax[i]
            dy = py - ay[i]
        else:
            t = (apx * abx[i] + apy * aby[i]) / ab_len2[i]

            if t < 0.0:
                dx = px - ax[i]
                dy = py - ay[i]
            elif t > 1.0:
                dx = px - bx[i]
                dy = py - by[i]
            else:
                closest_x = ax[i] + t * abx[i]
                closest_y = ay[i] + t * aby[i]

                dx = px - closest_x
                dy = py - closest_y

        dist2 = dx * dx + dy * dy

        if min_dist2 < 0.0 or dist2 < min_dist2:
            min_dist2 = dist2

    return math.sqrt(min_dist2)


@njit
def compare_routes_core_numba(
    runner_xs,
    runner_ys,
    ax,
    ay,
    bx,
    by,
    abx,
    aby,
    ab_len2,
    thresholds,
):
    """
    Wykonuje główną pętlę porównania tras dla danych w układzie metrycznym.
    Funkcja działa na tablicach NumPy i jest kompilowana przez Numbę.
    Zwraca wartości potrzebne do zbudowania wyniku końcowego.
    """
    total_distance_from_route = 0.0
    max_distance = 0.0

    threshold_counts = np.zeros(len(thresholds), dtype=np.int64)

    for i in range(len(runner_xs)):
        distance = distance_to_segments_numba(
            runner_xs[i],
            runner_ys[i],
            ax,
            ay,
            bx,
            by,
            abx,
            aby,
            ab_len2,
        )

        total_distance_from_route += distance

        if distance > max_distance:
            max_distance = distance

        for j in range(len(thresholds)):
            if distance <= thresholds[j]:
                threshold_counts[j] += 1

    mean_distance = total_distance_from_route / len(runner_xs)

    return mean_distance, max_distance, threshold_counts


def compare_routes_numba(
    reference_latitudes,
    reference_longitudes,
    runner_latitudes,
    runner_longitudes,
    thresholds_m=(5.0, 10.0, 20.0),
):
    """
    Porównuje trasę zawodnika z trasą referencyjną w wariancie Numba.
    Przekształca dane do układu EPSG:2180, buduje segmenty referencyjne
    i zwraca średnią odległość, maksymalne odchylenie oraz procent punktów
    w zadanych progach.
    """
    reference_xs, reference_ys = route_to_euclidean_numba(
        reference_latitudes,
        reference_longitudes,
    )

    runner_xs, runner_ys = route_to_euclidean_numba(
        runner_latitudes,
        runner_longitudes,
    )

    ax, ay, bx, by, abx, aby, ab_len2 = build_segments_numba(
        reference_xs,
        reference_ys,
    )

    thresholds = np.asarray(thresholds_m, dtype=np.float64)

    mean_distance, max_distance, threshold_counts = compare_routes_core_numba(
        runner_xs,
        runner_ys,
        ax,
        ay,
        bx,
        by,
        abx,
        aby,
        ab_len2,
        thresholds,
    )

    result = {
        "mean_distance_m": float(mean_distance),
        "max_distance_m": float(max_distance),
    }

    for i in range(len(thresholds)):
        key = f"within_{int(thresholds[i])}m_percent"
        result[key] = float(threshold_counts[i] / len(runner_xs) * 100.0)

    return result


@njit
def frechet_distance_xy_numba(reference_xs, reference_ys, runner_xs, runner_ys):
    """
    Liczy dyskretną odległość Frécheta dla punktów w układzie metrycznym.
    Wersja Numba używa pełnej macierzy DP, tak jak wersja listowa i NumPy.
    """

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

    return math.sqrt(ca[n - 1, m - 1])


def frechet_distance_numba(
    reference_latitudes,
    reference_longitudes,
    runner_latitudes,
    runner_longitudes,
):
    """
    Liczy dyskretną odległość Frécheta dla dwóch tras.
    Przekształca współrzędne do EPSG:2180, a właściwe obliczenia wykonuje w Numbie.
    """

    reference_xs, reference_ys = route_to_euclidean_numba(
        reference_latitudes,
        reference_longitudes,
    )

    runner_xs, runner_ys = route_to_euclidean_numba(
        runner_latitudes,
        runner_longitudes,
    )

    return float(
        frechet_distance_xy_numba(
            reference_xs,
            reference_ys,
            runner_xs,
            runner_ys,
        )
    )

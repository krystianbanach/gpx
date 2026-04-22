from gpx_list import (
    compute_route_stats,
    compute_time_stats,
    count_elevation_jumps,
    count_position_jumps,
)
from gpx_parser import MAX_DISTANCE_JUMP_M, MAX_ELEVATION_JUMP_M


def format_route_stats(stats):
    return (
        f" Dystans całkowity: {stats['total_distance'] / 1000.0:.2f} km\n"
        f" Suma podejść: {stats['elevation_gain']:.2f} m\n"
        f" Suma zejść: {stats['elevation_loss']:.2f} m"
    )


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

    for index, segment in enumerate(route_segments, start=1):
        if not segment:
            continue

        route_stats = compute_route_stats(segment)
        time_stats = compute_time_stats(segment)

        if time_stats:
            start_time, end_time, avg_speed, pace = time_stats
            lines.append(
                f" Segment {index}: punkty={len(segment)} | "
                f"dystans={route_stats['total_distance'] / 1000.0:.2f} km | "
                f"czas={end_time - start_time} | tempo={pace} | "
                f"podejścia={route_stats['elevation_gain']:.1f} m"
            )
        else:
            lines.append(
                f" Segment {index}: punkty={len(segment)} | "
                f"dystans={route_stats['total_distance'] / 1000.0:.2f} km | brak czasu"
            )

    return "\n".join(lines)


def format_gpx_analysis(track_points):
    return "\n".join([
        "ANALIZA DANYCH GPX",
        f"Liczba punktów: {len(track_points)}",
        f"Punkty bez danych o wysokości: {sum(1 for point in track_points if point.elevation is None)}",
        f"Punkty bez danych o czasie: {sum(1 for point in track_points if point.time is None)}",
        f"Skoki wysokości > {MAX_ELEVATION_JUMP_M} m: {count_elevation_jumps(track_points)}",
        f"Skoki pozycji > {MAX_DISTANCE_JUMP_M} m: {count_position_jumps(track_points)}",
    ])


def format_reference_stats(title, stats):
    return "\n".join([
        title,
        f" Średnia odległość od trasy: {stats['mean_distance_m']:.3f} m",
        f" Mediana odległości: {stats['median_distance_m']:.3f} m",
        f" Maksymalna odległość: {stats['max_distance_m']:.3f} m",
        f" Punkty w buforze 5 m: {stats['within_5m_percent']:.2f}%",
        f" Punkty w buforze 10 m: {stats['within_10m_percent']:.2f}%",
        f" Punkty w buforze 20 m: {stats['within_20m_percent']:.2f}%",
    ])


def build_full_report(
    route_stats,
    time_stats,
    route_segments,
    all_points,
    reference_list_stats=None,
    reference_numpy_stats=None,
):
    parts = [
        "STATYSTYKI TRASY",
        format_route_stats(route_stats),
        "",
        "STATYSTYKI CZASOWE",
        format_time_stats_segments(time_stats),
        "",
        format_segments_summary(route_segments),
        "",
        format_gpx_analysis(all_points),
    ]

    if reference_list_stats is not None:
        parts.extend([
            "",
            format_reference_stats("PORÓWNANIE Z TRASĄ REFERENCYJNĄ - LISTA", reference_list_stats),
        ])

    if reference_numpy_stats is not None:
        parts.extend([
            "",
            format_reference_stats("PORÓWNANIE Z TRASĄ REFERENCYJNĄ - NUMPY", reference_numpy_stats),
        ])

    return "\n".join(parts)


def save_stats_text(base_name, output_dir, report_text):
    import os

    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, f"{base_name}_stats.txt")

    with open(stats_file, "w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"Statystyki zapisane jako: {stats_file}")
    return stats_file
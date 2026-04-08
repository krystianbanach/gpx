from gpx_analysis import (
    compute_route_stats,
    compute_time_stats,
    count_elevation_jumps,
    count_position_jumps,
)
from gpx_parser import MAX_DISTANCE_JUMP_M, MAX_ELEVATION_JUMP_M


def format_route_stats(stats):
    return (
        f" Dystans całkowity: {stats['total_distance']:.2f} km\n"
        f" Suma podejść: {stats['elevation_gain']:.2f} m\n"
        f" Suma zejść: {stats['elevation_loss']:.2f} m"
    )


def format_time_stats(stats):
    if not stats:
        return 'Brak wystarczających danych czasowych.'

    start, end, avg_speed, pace = stats
    return (
        f" Czas rozpoczęcia: {start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f" Czas zakończenia: {end.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f" Czas trwania trasy: {end - start}\n"
        f" Średnia prędkość: {avg_speed:.2f} km/h\n"
        f" Tempo: {pace}"
    )


def format_time_stats_segments(stats):
    if not stats:
        return 'Brak wystarczających danych czasowych.'

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
    lines = ['SEGMENTY']
    for index, segment in enumerate(route_segments, start=1):
        if not segment:
            continue

        route_stats = compute_route_stats(segment)
        time_stats = compute_time_stats(segment)

        if time_stats:
            start, end, avg_speed, pace = time_stats
            lines.append(
                f" Segment {index}: punkty={len(segment)} | dystans={route_stats['total_distance']:.2f} km | "
                f"czas={end - start} | tempo={pace} | podejścia={route_stats['elevation_gain']:.1f} m"
            )
        else:
            lines.append(
                f" Segment {index}: punkty={len(segment)} | dystans={route_stats['total_distance']:.2f} km | brak czasu"
            )

    return '\n'.join(lines)


def format_gpx_analysis(track_points):
    lines = []
    lines.append('ANALIZA DANYCH GPX')
    lines.append(f'Liczba punktów: {len(track_points)}')
    lines.append(f'Punkty bez danych o wysokości: {sum(1 for pt in track_points if pt.elevation is None)}')
    lines.append(f'Punkty bez danych o czasie: {sum(1 for pt in track_points if pt.time is None)}')
    lines.append(f'Skoki wysokości > {MAX_ELEVATION_JUMP_M} m: {count_elevation_jumps(track_points)}')
    lines.append(f'Skoki pozycji > {MAX_DISTANCE_JUMP_M} m: {count_position_jumps(track_points)}')
    return '\n'.join(lines)


def build_full_report(route_stats, time_stats, route_segments, all_points):
    parts = []
    parts.append('STATYSTYKI TRASY')
    parts.append(format_route_stats(route_stats))
    parts.append('')
    parts.append('STATYSTYKI CZASOWE')
    parts.append(format_time_stats_segments(time_stats))
    parts.append('')
    parts.append(format_segments_summary(route_segments))
    parts.append('')
    parts.append(format_gpx_analysis(all_points))
    return '\n'.join(parts)


def save_stats_text(base_name, output_dir, report_text):
    import os

    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, f'{base_name}_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as file:
        file.write(report_text)
    print(f'Statystyki zapisane jako: {stats_file}')
    return stats_file

import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np


def parse_gpx_time(time_text):
    return datetime.fromisoformat(time_text.replace("Z", "+00:00")).timestamp()


def parser_py(file):
    ns = {
        "gpx": "http://www.topografix.com/GPX/1/1"
    }

    tree = ET.parse(file)
    root = tree.getroot()
    trackpoints = root.findall(".//gpx:trkpt", ns)

    latitudes = []
    longitudes = []
    elevations = []
    times = []

    for trkpt in trackpoints:
        lat_str = trkpt.get("lat")
        lon_str = trkpt.get("lon")
        ele_elem = trkpt.find("gpx:ele", ns)
        time_elem = trkpt.find("gpx:time", ns)

        if (
            lat_str is None
            or lon_str is None
            or ele_elem is None
            or not ele_elem.text
            or time_elem is None
            or not time_elem.text
        ):
            continue

        latitudes.append(float(lat_str))
        longitudes.append(float(lon_str))
        elevations.append(float(ele_elem.text))
        times.append(parse_gpx_time(time_elem.text))

    return latitudes, longitudes, elevations, times


def parser_np(file):
    ns = {
        "gpx": "http://www.topografix.com/GPX/1/1"
    }

    tree = ET.parse(file)
    root = tree.getroot()
    trackpoints = root.findall(".//gpx:trkpt", ns)

    n = len(trackpoints)

    latitudes = np.empty(n, dtype=np.float64)
    longitudes = np.empty(n, dtype=np.float64)
    elevations = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)

    i = 0

    for trkpt in trackpoints:
        lat_str = trkpt.get("lat")
        lon_str = trkpt.get("lon")
        ele_elem = trkpt.find("gpx:ele", ns)
        time_elem = trkpt.find("gpx:time", ns)

        if (
            lat_str is None
            or lon_str is None
            or ele_elem is None
            or not ele_elem.text
            or time_elem is None
            or not time_elem.text
        ):
            continue

        latitudes[i] = float(lat_str)
        longitudes[i] = float(lon_str)
        elevations[i] = float(ele_elem.text)
        times[i] = parse_gpx_time(time_elem.text)

        i += 1

    return latitudes[:i], longitudes[:i], elevations[:i], times[:i]

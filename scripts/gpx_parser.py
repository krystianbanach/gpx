import xml.etree.ElementTree as ET
import numpy as np


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

    for trkpt in trackpoints:
        lat_str = trkpt.get("lat")
        lon_str = trkpt.get("lon")
        ele_elem = trkpt.find("gpx:ele", ns)

        if (
            lat_str is None
            or lon_str is None
            or ele_elem is None
            or not ele_elem.text
        ):
            continue

        latitudes.append(float(lat_str))
        longitudes.append(float(lon_str))
        elevations.append(float(ele_elem.text))

    return latitudes, longitudes, elevations


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

    i = 0

    for trkpt in trackpoints:
        lat_str = trkpt.get("lat")
        lon_str = trkpt.get("lon")
        ele_elem = trkpt.find("gpx:ele", ns)

        if (
            lat_str is None
            or lon_str is None
            or ele_elem is None
            or not ele_elem.text
        ):
            continue

        latitudes[i] = float(lat_str)
        longitudes[i] = float(lon_str)
        elevations[i] = float(ele_elem.text)

        i += 1

    return latitudes[:i], longitudes[:i], elevations[:i]
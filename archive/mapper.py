import numpy as np
import folium
from folium.plugins import HeatMap
import webbrowser
import pandas as pd
import os
import time
import tempfile
import statistics
import math
from scipy import stats

map_cbg_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_geographic_data.csv'))


def display_data(data, map_lat=38.8300, map_long=-77.2800, zoom_level=11, labels=[]):  # [[lat, long, weight], [lat, long, weight], ... [lat, long, weight]]
    # create map
    folium_map = folium.Map([map_lat, map_long], tiles='stamentoner', zoom_start=11)

    for idx, topic in enumerate(data):
        h = HeatMap(topic, show=False)
        if labels:
            h.layer_name = labels[idx]
        else:
            h.layer_name = 'topic_{}'.format(idx)
        folium_map.add_child(h)
    
    folium.LayerControl(collapsed=False).add_to(folium_map)

    temp_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map.html'))
    folium_map.save(temp_path)
    webbrowser.open(temp_path)
    time.sleep(1)
    os.remove(temp_path)


def cbgs_to_map(input_data, labels=[]):  # [[(0.20465624, '510594825022'), (0.12849134, '510594826011'), ...], [(0.13041233, '510594825022'), (0.10988416, '510594826011'), ...], ...]
    mapping_data = []
    for topic in input_data:
        topic_data = []
        for tup in topic:
            row = map_cbg_data.loc[map_cbg_data.census_block_group == int(tup[0])]
            topic_data.append([float(row.latitude), float(row.longitude), float(tup[1]) * 1000])
        mapping_data.append(topic_data)
    display_data(mapping_data, labels=labels)

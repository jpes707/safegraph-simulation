import numpy as np
import folium
from folium.plugins import HeatMap
import webbrowser
import pandas as pd
import os
import time

cbg_data = pd.read_csv(r'E:\safegraph-simulation\safegraph-data\safegraph_open_census_data\metadata\cbg_geographic_data.csv')

def display_data(data):  # [[lat, long, weight], [lat, long, weight], ... [lat, long, weight]]
    # create map
    folium_map = folium.Map([38.8300, -77.2800], tiles='stamentoner', zoom_start=11)

    for idx, topic in enumerate(data):
        h = HeatMap(topic)
        h.layer_name = 'topic_{}'.format(idx)
        folium_map.add_child(h)
    
    folium.LayerControl(collapsed=False).add_to(folium_map)

    temp_path = os.path.join('.', 'tmpfile.html')
    folium_map.save(temp_path)
    webbrowser.open(temp_path)
    time.sleep(1)
    os.remove(temp_path)


def lda_to_map(lda_data):  # lda.top_topics(lda_corpus)
    mapping_data = []
    for topic in lda_data:
        topic_data = []
        for tup in topic[0]:
            row = cbg_data.loc[cbg_data.census_block_group == int(tup[1])]
            topic_data.append([float(row.latitude), float(row.longitude), float(tup[0]) * 1000])
        mapping_data.append(topic_data)
    display_data(mapping_data)

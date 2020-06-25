import os
import pandas as pd
from mapper import cbgs_to_map
from lda import lda

# This program assumes dictionaries iterate in insertion order, requires python3.7+

STATE_ABBR = input('State abbreviation (example: VA): ')
if STATE_ABBR == '':
    STATE_ABBR = 'VA'
LOCALITY = input('Locality (example: Fairfax): ')  # County, Borough, or Parish
if LOCALITY == '':
    LOCALITY = 'Fairfax'
TABLE_ID = input('Census Table ID (example: B19001): ')
if TABLE_ID == '':
    TABLE_ID = 'B19001'
WEEK = input('Week for LDA model (example: 2020-05-25): ')
if WEEK == '':
    WEEK = '2020-06-01'

if STATE_ABBR == 'AK':
    LOCALITY_TYPE = 'borough'
elif STATE_ABBR == 'LA':
    LOCALITY_TYPE = 'parish'
else:
    LOCALITY_TYPE = 'county'

locality_name = (LOCALITY + ' ' + LOCALITY_TYPE).title()

print()
print('--------------------------------------------------------------------------------------------')
print('{}, {}'.format(locality_name, STATE_ABBR.upper()))

table_shells = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'other-data', 'ACS2018_Table_Shells.csv'))
iter_count = 0
description_dict = {}
for idx, row in table_shells[table_shells['Table ID'] == TABLE_ID.upper()].iterrows():
    s = str(row['Stub'])
    if iter_count == 0:
        description = s
    elif iter_count == 1:
        universe = s[11:]
    else:
        key = str(row['UniqueID'])
        key = key[:6] + 'e' + str(int(key[7:]))
        if ':' not in s:
            description_dict[key] = [s, {}]
    iter_count += 1
print(description)
print('Universe: {}'.format(universe))
print('--------------------------------------------------------------------------------------------')
print()

print('Finding FIPS prefix...')

fips_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_fips_codes.csv'))
locality_data = str(fips_data.loc[(fips_data.state == STATE_ABBR.upper()) & (fips_data.county == locality_name)]).split()[7:9]
fips_prefix = (str(locality_data[0]).zfill(2) + str(locality_data[1]).zfill(3)).replace(' ', '')

print('Found FIPS prefix: {}, obtaining set of applicable CBGs'.format(fips_prefix))

cbgs = set()
cbg_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_geographic_data.csv'))
for idx, row in cbg_data.iterrows():
    check_str = str(int(row['census_block_group'])).zfill(12)
    if check_str.startswith(fips_prefix):
        cbgs.add(check_str)

print('Obtained set of applicable CBGs, gathering census data...')

census_cbg_name = 'cbg_' + TABLE_ID[:3].lower() + '.csv'
census_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', census_cbg_name))
for idx, row in census_data.iterrows():
    check_str = str(int(row['census_block_group'])).zfill(12)
    if check_str in cbgs:
        for key in description_dict:
            pop = float(row[key])
            description_dict[key][1][check_str] = pop

'''
print('Gathered census data, displaying visualization')

visual_list = [list(description_dict[key][1]) for key in description_dict]
labels = [description_dict[key][0] for key in description_dict]
cbgs_to_map(visual_list, labels)
'''

print('Gathered census data, running LDA...')

area = locality_name.lower().replace(' ', '-') + '-' + STATE_ABBR.lower()
lda_model, lda_corpus = lda(area, WEEK)
top_topics = [elem[1] for elem in lda_model.show_topics(formatted=False)]

print('LDA complete, weighting categorical values...')

topic_dictionary = {}
for idx, topic in enumerate(top_topics):  # [(0.20465624, '510594825022'), (0.12849134, '510594826011'), ...], [(0.13041233, '510594825022'), (0.10988416, '510594826011'), ...], ...
    topic_key = 'topic_{}'.format(idx)
    numerator = [0] * len(description_dict)
    denominator = [0] * len(description_dict)
    for tup in topic:
        weight = tup[1]
        cbg = tup[0]
        cur_idx = 0
        for key in description_dict:
            if cbg in description_dict[key][1]:
                numerator[cur_idx] += description_dict[key][1][cbg] * weight
                denominator[cur_idx] += weight
            else:
                print(cbg, weight)
            cur_idx += 1
    print(numerator)
    print(denominator)
    topic_value = [numerator[i]/denominator[i] for i in range(len(description_dict))]
    topic_dictionary[topic_key] = topic_value

print('Categorical weighting complete, creating CSV...')

csv = ',"' + '","'.join(['{} ({})'.format(key, description_dict[key][0]) for key in description_dict]) + '",\n'
for key in topic_dictionary:
    csv += key + ',' + ','.join(str(val) for val in topic_dictionary[key]) + ',\n'
folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demographics-outputs')
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demographics-outputs', area + '-' + WEEK + '.csv')
f = open(csv_path, 'w')
f.write(csv)
f.close()

print('CSV created at {}'.format(csv_path))
print('Complete! Displaying visualization in web browser...')
cbgs_to_map(top_topics, list(topic_dictionary.keys()))

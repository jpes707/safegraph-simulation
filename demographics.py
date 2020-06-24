import os
import pandas as pd
from mapper import cbgs_to_map

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

if STATE_ABBR == 'AK':
    LOCALITY_TYPE = 'borough'
elif STATE_ABBR == 'LA':
    LOCALITY_TYPE = 'parish'
else:
    LOCALITY_TYPE = 'county'

locality_name = LOCALITY[0].upper() + LOCALITY[1:].lower() + ' ' + LOCALITY_TYPE[0].upper() + LOCALITY_TYPE[1:].lower()

print()
print('{}, {}'.format(locality_name, STATE_ABBR.upper()))

table_shells = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'other-data', 'ACS2018_Table_Shells.csv'))
iter_count = 0
description_dict = {}
for idx, row in table_shells[table_shells['Table ID'] == TABLE_ID.upper()].iterrows():
    if iter_count == 0:
        description = str(row['Stub'])
    elif iter_count == 1:
        universe = str(row['Stub'])[11:]
    else:
        key = str(row['UniqueID'])
        key = key[:6] + 'e' + str(int(key[7:]))
        description_dict[key] = [str(row['Stub']).replace(':',''), set()]
    iter_count += 1
print(description)
print('Universe: {}'.format(universe))
print()

print('Finding FIPS prefix')

fips_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_fips_codes.csv'))
locality_data = str(fips_data.loc[(fips_data.state == STATE_ABBR.upper()) & (fips_data.county == locality_name)]).split()[7:9]
fips_prefix = (str(locality_data[0]).zfill(2) + str(locality_data[1]).zfill(3)).replace(' ', '')

print('Found FIPS prefix: {}, obtaining set of applicable CBGs'.format(fips_prefix))

cbgs = set()
cbg_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_geographic_data.csv'))
for idx, row in cbg_data.iterrows():
    check_str = str(int(row['census_block_group']))
    if check_str.zfill(12).startswith(fips_prefix):
        cbgs.add(check_str)

print('Obtained set of applicable CBGs, gathering census data')

census_cbg_name = 'cbg_' + TABLE_ID[:3].lower() + '.csv'
census_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', census_cbg_name))
demographics_data = census_data.loc[census_data.census_block_group.astype(str).isin(cbgs)]
for idx, row in demographics_data.iterrows():
    total = -1
    for key in description_dict:
        pop = float(row[key])
        if total == -1:
            if pop == 0:
                break
            description_dict[key][1].add((pop, str(int(row['census_block_group']))))
            total = pop
        else:
            description_dict[key][1].add((pop / total, str(int(row['census_block_group']))))

print('Gathered census data, displaying visualization')

visual_list = [list(description_dict[key][1]) for key in description_dict]
labels = [description_dict[key][0] for key in description_dict]
cbgs_to_map(visual_list, labels)

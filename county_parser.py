import pandas as pd
import requests
import os
import json
import shutil
import gzip
import numpy as np
from distfit import distfit

MINIMUM_RAW_VISITOR_COUNT = 30

WEEK = input('Week (example: 2020-05-25): ')
STATE_ABBR = input('State abbreviation (example: VA): ')
LOCALITY = input('Locality (example: Fairfax): ') # County, Borough, or Parish

if STATE_ABBR == 'AK':
    LOCALITY_TYPE = 'borough'
elif STATE_ABBR == 'LA':
    LOCALITY_TYPE = 'parish'
else:
    LOCALITY_TYPE = 'county'


buckets = ['1-5', '6-20', '21-60', '61-240', '241-960']
def get_dwell_distribution(dwell_dictionary):
    nums = list(dwell_dictionary.values())
    print(nums)

    filled_arr = np.empty([1, 1])
    for index, bucket in enumerate(buckets):
        lower, upper = map(int, bucket.split('-'))
        filled_arr = np.concatenate((filled_arr, np.random.uniform(low=lower, high=upper, size=(int(nums[index]), 1))))

    filled_arr = filled_arr[~np.isnan(filled_arr)]

    dist = distfit()
    dist.fit_transform(filled_arr)
    dist_name = dist.model['name']
    
    if len(dist.model['arg']) < 1:
        return (dist_name, dist.model['loc'], dist.model['scale'])
    elif len(dist.model['arg']) == 2:
        return (dist_name, float((dist.model['arg'])[0]), float((dist.model['arg'])[1]), dist.model['loc'], dist.model['scale'])
    else:
        return (dist_name, float((dist.model['arg'])[0]), dist.model['loc'], dist.model['scale'])


res = requests.get('https://www.zillow.com/browse/homes/{}/{}-{}/'.format(STATE_ABBR.lower(), LOCALITY.lower(), LOCALITY_TYPE), headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36', 'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9', 'referer': 'https://www.google.com/'}).text
res = res[res.find('<ul>'):]
res = res[:res.find('</ul>')]
zip_code_set = {str(s[:5]) for s in res.split('">')[1:]}
print('Zip codes:', zip_code_set)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-weekly-patterns.csv'.format(WEEK))
if not os.path.exists(data_path):
    print('Extracting weekly data...')
    os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK)))
    with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns.csv.gz'.format(WEEK)), 'rb') as f_in:
        with open(data_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
print('Reading data...')
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-weekly-patterns.csv'.format(WEEK)), error_bad_lines=False)
print('Processing data...')
county_data = data[(data.raw_visitor_counts >= MINIMUM_RAW_VISITOR_COUNT) & (data.postal_code.astype(str).isin(zip_code_set))]
print('Running distributions...')
county_data['dwell_distribution'] = '()'
for idx, row in county_data.iterrows():
    county_data.loc[idx,'dwell_distribution'] = str(get_dwell_distribution(json.loads(row.bucketed_dwell_times)))
print('Writing data...')
print(county_data)
county_data.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-{}-{}-{}.csv'.format(LOCALITY.replace(' ', '-').lower(), LOCALITY_TYPE, STATE_ABBR.lower(), WEEK)))
print('Complete!')

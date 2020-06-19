import pandas as pd
import requests
import os

WEEK = input('Week (example: 2020-05-25): ')
STATE_ABBR = input('State abbreviation (example: VA): ')
LOCALITY = input('Locality (example: Fairfax): ')  # County, Borough, or Parish

if STATE_ABBR == 'AK':
    LOCALITY_TYPE = 'borough'
elif STATE_ABBR == 'LA':
    LOCALITY_TYPE = 'parish'
else:
    LOCALITY_TYPE = 'county'

res = requests.get('https://www.zillow.com/browse/homes/{}/{}-{}/'.format(STATE_ABBR.lower(), LOCALITY.lower(), LOCALITY_TYPE), headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36', 'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9', 'referer': 'https://www.google.com/'}).text
res = res[res.find('<ul>'):]
res = res[:res.find('</ul>')]
zip_code_set = {str(s[:5]) for s in res.split('">')[1:]}
print('Zip codes:', zip_code_set)

print('Reading data...')
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-weekly-patterns.csv'.format(WEEK)), error_bad_lines=False)
print('Processing data...')
county_data = data[data.postal_code.astype(str).isin(zip_code_set)]
print('Writing data...')
print(county_data)
county_data.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-{}-{}-{}.csv'.format(LOCALITY.lower(), LOCALITY_TYPE, STATE_ABBR.lower(), WEEK)))
print('Complete!')

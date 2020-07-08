import pandas as pd
import os

WEEK = input('Week (example: 2020-05-25): ')
STATE_ABBR = input('State abbreviation (example: VA): ')

print('Reading data...')
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-weekly-patterns.csv'.format(WEEK)), error_bad_lines=False)
print('Processing data...')
state_data = data[data.region == STATE_ABBR]
print('Writing data...')
state_data.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), 'state-{}-{}.csv'.format(STATE_ABBR.lower(), WEEK)))
print('Complete!')

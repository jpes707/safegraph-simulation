import pandas as pd

WEEK = '2020-06-01'
STATE_ABBR =  'RI'

print('Reading data...')
data = pd.read_csv(r'./safegraph-data/safegraph_weekly_patterns_v2/main-file/{}-weekly-patterns.csv/{}-weekly-patterns.csv'.format(WEEK, WEEK), error_bad_lines=False)
print('Processing data...')
state_data = data[data.region == STATE_ABBR]
print('Writing data...')
state_data.to_csv(r'./safegraph-data/safegraph_weekly_patterns_v2/main-file/{}-weekly-patterns.csv/state-{}-{}.csv'.format(WEEK, STATE_ABBR.lower(), WEEK))
print('Complete!')

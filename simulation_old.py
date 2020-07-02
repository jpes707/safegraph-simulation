import json
import numpy
import os
import pickle
import statistics
import math
import pandas as pd
from lda import lda

STATE_ABBR = input('State abbreviation (default: VA): ')
if STATE_ABBR == '':
    STATE_ABBR = 'VA'
LOCALITY = input('Locality (default: Fairfax): ')  # County, Borough, or Parish
if LOCALITY == '':
    LOCALITY = 'Fairfax'
# TABLE_ID = input('Census Table ID (default: B19001): ')
# if TABLE_ID == '':
#     TABLE_ID = 'B19001'
WEEK = input('Week (default: 2019-10-28): ')
if WEEK == '':
    WEEK = '2019-10-28'

if STATE_ABBR == 'AK':
    LOCALITY_TYPE = 'borough'
elif STATE_ABBR == 'LA':
    LOCALITY_TYPE = 'parish'
else:
    LOCALITY_TYPE = 'county'

locality_name = (LOCALITY + ' ' + LOCALITY_TYPE).title()
full_name = '{}, {}'.format(locality_name, STATE_ABBR.upper())
area = locality_name.lower().replace(' ', '-') + '-' + STATE_ABBR.lower()

print('Running LDA...')

lda_model, lda_corpus, place_to_bow, place_to_counts, lda_dictionary, usable_data = lda(area, WEEK)
top_topics = [elem[1] for elem in lda_model.show_topics(formatted=False)]
cbgs = {tup[0] for topic in top_topics for tup in topic}
num_topics = len(lda_model.get_topics())
cbgs_to_agents = {}  # cbg: {agent id, agent id, ...}
agents = {}  # agent_id: (topic, infected_status)
place_current_visitors = {key: set() for key in place_to_bow}  # place id: {agent id, agent id, ...}
active_agent_ids = {}  # expiration time: {agent id, agent id, ...}
inactive_agent_ids = set()

'''print('Mapping topics to places...')

topic_to_places = [list() for i in range(num_topics)]
for key in place_to_bow:
    for tup in lda_model[place_to_bow[key]]:
        topic_to_places[tup[0]].append((tup[1] * place_to_counts[key], key))
for idx, elem in enumerate(topic_to_places):
    total_weights = 0
    for tup in elem:
        total_weights += tup[0]
    topic_to_places[idx] = sorted([(tup[0] / total_weights, tup[1]) for tup in elem], reverse=True)'''

cbg_topics_dict = {}
lda_matrix = lda_model.get_topics()
lda_dict = dict(lda_dictionary)
for key in lda_dict:
    cbg_topics = []
    for idx in range(len(lda_matrix)):
        cbg_topics.append((idx, lda_matrix[idx][key]))
    cbg_topics_dict[lda_dict[key]] = sorted(cbg_topics, key=lambda x: x[1], reverse=True)

print('Reading social distancing data...')

social_distancing_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_social_distancing_metrics', WEEK[:4], WEEK[5:7], WEEK[8:], '{}-social-distancing.csv'.format(WEEK)), error_bad_lines=False)
# cbg_data = {}  # cbg: non_home_cbg_population
total_population = 0
topic_population_counts = [0] * num_topics
for idx, row in social_distancing_data.iterrows():
    check_str = str(int(row['origin_census_block_group'])).zfill(12)
    if check_str in cbgs:
        # cbg_population = int(row['device_count'])  # candidate_device_count for actual safegraph population any time
        # chance_to_leave = 1 - (int(row['completely_home_device_count']) / cbg_population)
        # cbg_data[check_str] = (cbg_population, chance_to_leave)
        # cbg_data[check_str] = cbg_population - row['completely_home_device_count']
        # total_population += cbg_population
        #topics = lda_model[check_str]
        non_home_cbg_population = int(row['device_count']) - int(row['completely_home_device_count'])
        total_population += non_home_cbg_population
        for tup in cbg_topics_dict[check_str]:
            if tup[0] == 0:
                print(non_home_cbg_population, tup[1], tup[1] * non_home_cbg_population)
            topic_population_counts[tup[0]] += tup[1] * non_home_cbg_population

'''
proportion = total_population / sum(topic_population_counts)
for idx in range(num_topics):
    topic_population_counts[idx] *= proportion

print(sum(topic_population_counts), total_population)
print(topic_population_counts)
'''

'''
cbg_proportions = []
total_proportions = 0
for key in cbg_data:
    current_proportion = cbg_data[key][0] / total_population
    total_proportions += current_proportion
    cbg_proportions.append((total_proportions, key))
print(cbg_proportions)
'''

exit()

extrapolation_set_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'other-data', 'ACS2018_Table_Shells.csv')
extrapolation_set = pickle.load(open(extrapolation_set_file_path, 'rb'))
extrapolation_set.add(259.6057407338894)


def dwell_time_distribution(bucketed_string):
    json_obj = json.loads(bucketed_string)
    obj_list = [json_obj[key] for key in json_obj]
    points = [(12.5, sum(obj_list[1:])), (40, sum(obj_list[2:])), (150, sum(obj_list[3:]))]
    x = numpy.array([tup[0] for tup in points])
    y = numpy.array([tup[1] for tup in points])
    regression_coeffs = list(numpy.polyfit(numpy.log(x), y, 1))
    a, b = regression_coeffs[0], regression_coeffs[1]
    x_intercept = math.e ** (-b / a)
    trials = 0
    while ((a * math.log(240) + b > 0) != (bool(obj_list[4])) or (x_intercept > 960)) and trials < 100:
        print('redoing')
        points.append((statistics.median(extrapolation_set) if obj_list[4] else 240, obj_list[4]))
        x = numpy.array([tup[0] for tup in points])
        y = numpy.array([tup[1] for tup in points])
        regression_coeffs = list(numpy.polyfit(numpy.log(x), y, 1))
        a, b = regression_coeffs[0], regression_coeffs[1]
        x_intercept = math.e ** (-b / a)
        trials += 1
    print(points)
    print('y = {}ln(x) + {}'.format(a, b))
    if trials == 100:
        x_intercept = 840
    elif a * math.log(240) + b > 0:
        final_bucket_intercept = math.e ** ((obj_list[4] - b) / a)
        # extrapolation_set.add(final_bucket_intercept)
    def f(x):
        # return (((a * math.log(x_intercept) + b) - (a * math.log(5) + b)) * x + (a * math.log(5) + b) - b) / a
        # return math.e ** (math.log(x_intercept) * x - math.log(5) * (x - 1))
        return x_intercept ** x * 5 ** (1 - x)
    return f


g = dwell_time_distribution('{"<5":0,"5-20":17,"21-60":16,"61-240":60,">240":137}')
print(g(1))

exit()
pickle.dump(extrapolation_set, open(extrapolation_set_file_path, 'wb'))

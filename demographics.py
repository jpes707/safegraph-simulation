import os
import pandas as pd
from mapper import cbgs_to_map
from lda import lda
from scipy.stats import chi2_contingency, chisquare

# This program assumes dictionaries iterate in insertion order, requires python3.7+

# Common census codes
# B01001 - age
# B02001 - race
# B19001 - income brackets

ALPHA = 0.05  # for chi-squared p-test

STATE_ABBR = input('State abbreviation (default: VA): ')
if STATE_ABBR == '':
    STATE_ABBR = 'VA'
LOCALITY = input('Locality (default: Fairfax): ')  # County, Borough, or Parish
if LOCALITY == '':
    LOCALITY = 'Fairfax'
# TABLE_ID = input('Census Table ID (default: B19001): ')
# if TABLE_ID == '':
#     TABLE_ID = 'B19001'
WEEK = input('Week for LDA model (default: 2020-06-01): ')
if WEEK == '':
    WEEK = '2020-06-01'
NUM_TOPICS = input('Number of topics to analyze (default: 20): ')
if NUM_TOPICS == '':
    NUM_TOPICS = 20
else:
    NUM_TOPICS = int(NUM_TOPICS)

if STATE_ABBR == 'AK':
    LOCALITY_TYPE = 'borough'
elif STATE_ABBR == 'LA':
    LOCALITY_TYPE = 'parish'
else:
    LOCALITY_TYPE = 'county'

locality_name = (LOCALITY + ' ' + LOCALITY_TYPE).title()
full_name = '{}, {}'.format(locality_name, STATE_ABBR.upper())

area = locality_name.lower().replace(' ', '-') + '-' + STATE_ABBR.lower()
lda_model, lda_corpus = lda(area, WEEK)
top_topics = [elem[1] for elem in lda_model.show_topics(formatted=False, num_topics=NUM_TOPICS)]
cbgs = {tup[0] for topic in top_topics for tup in topic}

print()
for TABLE_ID in ['B01001', 'B02001', 'B19001']:
    print('--------------------------------------------------------------------------------------------')
    print(full_name)

    table_shells = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'other-data', 'ACS2018_Table_Shells.csv'))
    iter_count = 0
    description_dict = {}
    reverse_description_dict = {}
    for idx, row in table_shells[table_shells['Table ID'] == TABLE_ID.upper()].iterrows():
        s = str(row['Stub']).strip()
        if iter_count == 0:
            alternate_label_dict = {'B01001': 'AGE'}
            if TABLE_ID.upper() in alternate_label_dict:
                description = alternate_label_dict[TABLE_ID.upper()]
            else:
                description = s
        elif iter_count == 1:
            universe = s[11:]
        else:
            unique_id = str(row['UniqueID'])
            key = (unique_id[:6] + 'e' + str(int(unique_id[7:])),)
            if ':' not in s:
                if s in reverse_description_dict:
                    old_key = reverse_description_dict[s]
                    del description_dict[old_key]
                    key = tuple(old_key + key)
                    reverse_description_dict[s] = key
                    description_dict[key] = [s, {}]
                else:
                    reverse_description_dict[s] = key
                    description_dict[key] = [s, {}]
        iter_count += 1
    print(description)
    print('Universe: {}'.format(universe))
    print('Categories: {}'.format('"' + '", "'.join([description_dict[key][0] for key in description_dict]) + '"'))
    print('--------------------------------------------------------------------------------------------')
    print()

    print('Gathering census data...')

    census_cbg_name = 'cbg_' + TABLE_ID[:3].lower() + '.csv'
    census_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', census_cbg_name))
    for idx, row in census_data.iterrows():
        check_str = str(int(row['census_block_group'])).zfill(12)
        if check_str in cbgs:
            for tup in description_dict:
                for key in tup:
                    if check_str in description_dict[tup][1]:
                        description_dict[tup][1][check_str] += float(row[key])
                    else:
                        description_dict[tup][1][check_str] = float(row[key])

    print('Census data gathered, weighting categorical values...')

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
                cur_idx += 1
        topic_value = [numerator[i]/denominator[i] for i in range(len(description_dict))]
        topic_dictionary[topic_key] = topic_value

    print('Categorical weighting complete, running chi-squared test of independence...')

    print()
    g, p, degrees_of_freedom, expected_matrix = chi2_contingency(list(topic_dictionary.values()))
    if p > ALPHA:
        print('Chi-squared independence test p-value is {} > alpha value of {} => not statistically significant. This means {} does not influence topic distributions in {}. Try again with another locality or census group.'.format(p, ALPHA, description, full_name))
        print()
        continue
    print('Chi-squared independence test p-value is {} < alpha value of {} => statistically significant. This means {} influences topic distributions in {}! Running individual chi-squared tests for each topic...'.format(p, ALPHA, description, full_name))
    print('Dividing alpha ({}) by the number of topics ({}) for Bonferroni correction...'.format(ALPHA, len(topic_dictionary)))
    new_alpha = ALPHA / len(topic_dictionary)
    print('The new alpha value is {}.'.format(new_alpha))
    print()
    influenced_topics = list()
    current_idx = 0
    for key in topic_dictionary:
        chisq, p = chisquare(topic_dictionary[key], f_exp=expected_matrix[current_idx])
        if p > new_alpha:
            print('Chi-squared goodness of fit test p-value for {} is {} > alpha value of {} => not statistically significant.'.format(key, p, new_alpha))
        else:
            influenced_topics.append(key)
            print('Chi-squared goodness of fit test p-value for {} is {} < alpha value of {} => statistically significant!'.format(key, p, new_alpha))
        current_idx += 1
    print()
    print('Summary: {} affects {}/{} topics ({}%) in {}.'.format(description, len(influenced_topics), len(topic_dictionary), str(len(influenced_topics) / len(topic_dictionary) * 100)[:6], full_name))
    print('Influenced topics: {}'.format(', '.join(influenced_topics)))
    print()

print('Complete! Displaying visualization in web browser...')
print()
cbgs_to_map(top_topics, list(topic_dictionary.keys()))

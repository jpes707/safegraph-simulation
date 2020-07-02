import pandas as pd
import gensim
import json
import random
import os

MINIMUM_RAW_VISITOR_COUNT = 30

STATE_ABBR = input('State abbreviation (default: VA): ')
if STATE_ABBR == '':
    STATE_ABBR = 'VA'
LOCALITY = input('Locality (default: Fairfax): ')  # County, Borough, or Parish
if LOCALITY == '':
    LOCALITY = 'Fairfax'
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

data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-{}.csv'.format(area, WEEK)), error_bad_lines=False)
usable_data = data[(data.raw_visitor_counts >= MINIMUM_RAW_VISITOR_COUNT) & (data.visitor_home_cbgs != '{}')]

cbg_to_places = {}
for row in usable_data.itertuples():
    place_id = str(row.safegraph_place_id)
    cbgs = json.loads(row.visitor_home_cbgs)
    lda_words = []
    for cbg in cbgs:
        if cbg not in cbg_to_places:
            cbg_to_places[cbg] = []
        cbg_frequency = random.randint(2, 4) if cbgs[cbg] == 4 else cbgs[cbg]
        cbg_to_places[cbg].extend([place_id] * cbg_frequency)

cbg_ids = cbg_to_places.keys()
cbg_id_set = set(cbg_ids)
lda_documents = cbg_to_places.values()

lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents)
lda_corpus = [lda_dictionary.doc2bow(cbg) for cbg in lda_documents]
cbg_to_bow = dict(zip(cbg_ids, lda_corpus))
lda_model = gensim.models.HdpModel(lda_corpus, id2word=lda_dictionary).suggested_lda_model()

cbgs_to_topics = dict(zip(cbg_ids, list(lda_model.get_document_topics(lda_corpus, minimum_probability=0))))
topics_to_pois = [elem[1] for elem in lda_model.show_topics(formatted=False, num_topics=150)]

print('Reading social distancing data...')

social_distancing_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_social_distancing_metrics', WEEK[:4], WEEK[5:7], WEEK[8:], '{}-social-distancing.csv'.format(WEEK)), error_bad_lines=False)
cbgs_to_populations = {}
total_population = 0
for idx, row in social_distancing_data.iterrows():
    check_str = str(int(row['origin_census_block_group'])).zfill(12)
    if check_str in cbg_id_set:
        non_home_cbg_population = int(row['device_count']) - int(row['completely_home_device_count'])
        total_population += non_home_cbg_population
        cbgs_to_populations[check_str] = non_home_cbg_population

print('Establishing probabilities...')

cbg_proportions = []
total_proportions = 0
for key in cbgs_to_populations:
    current_proportion = cbgs_to_populations[key] / total_population
    total_proportions += current_proportion
    cbg_proportions.append((total_proportions, key))

print('Preparing simulation...')

agents = {}  # agent_id: [topic, infected_status]
cbgs_to_agents = {cbg: set() for cbg in cbg_ids}  # cbg: {agent id, agent id, ...}
inactive_agent_ids = set()
active_agent_ids = {}  # expiration time: {agent id, agent id, ...}
place_current_visitors = {poi: set() for poi_list in lda_documents for poi in poi_list}  # place id: {agent id, agent id, ...}

for iteration in range(total_population):
    rand = random.random()
    for elem in cbg_proportions:
        if rand < elem[0]:
            current_cbg = elem[1]
            agent_id = 'agent_' + str(iteration)
            agent_topic = 5  # change soon
            agents[agent_id] = [agent_topic, 0]
            cbgs_to_agents[current_cbg].add(agent_id)
            inactive_agent_ids.add(agent_id)
            break

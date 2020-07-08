import pandas as pd
import gensim
import json
import random
import os
import numpy
import time
import math

NUM_TOPICS = 50
MINIMUM_RAW_VISITOR_COUNT = 30
SIMULATION_DAYS = 10
SIMULATION_HOURS_PER_DAY = 16
SIMULATION_TICKS_PER_HOUR = 60
PROPORTION_OF_POPULATION = 0.2  # 0.2 => 20% of the actual population is simulated

start_time = time.time()
total_simulation_time = SIMULATION_HOURS_PER_DAY * SIMULATION_TICKS_PER_HOUR


def adj_sig_figs(a, n=3):  # truncate the number a to have n significant figures
    if not a:
        return 0
    dec = int(math.log10(abs(a)) // 1)
    z = int((a * 10 ** (n - 1 - dec) + 0.5) // 1) / 10 ** (n - 1 - dec)
    return str(z) if z % 1 else str(int(z))


def print_elapsed_time():
    print('Total time elapsed: {}s'.format(adj_sig_figs(time.time() - start_time, 4)))


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

cbgs_to_places = {}
for row in usable_data.itertuples():
    place_id = str(row.safegraph_place_id)
    cbgs = json.loads(row.visitor_home_cbgs)
    lda_words = []
    for cbg in cbgs:
        if cbg not in cbgs_to_places:
            cbgs_to_places[cbg] = []
        cbg_frequency = random.randint(2, 4) if cbgs[cbg] == 4 else cbgs[cbg]
        cbgs_to_places[cbg].extend([place_id] * cbg_frequency)

cbg_ids = list(cbgs_to_places.keys())
cbg_id_set = set(cbg_ids)
lda_documents = list(cbgs_to_places.values())
poi_set = {poi for poi_list in lda_documents for poi in poi_list}
poi_count = len(poi_set)

lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents)
lda_corpus = [lda_dictionary.doc2bow(cbg) for cbg in lda_documents]
cbg_to_bow = dict(zip(cbg_ids, lda_corpus))
lda_model = gensim.models.LdaModel(lda_corpus, num_topics=NUM_TOPICS, id2word=lda_dictionary)

cbgs_to_topics = dict(zip(cbg_ids, list(lda_model.get_document_topics(lda_corpus, minimum_probability=0))))
topics_to_pois = [[[tup[0] for tup in givens[1]], [tup[1] for tup in givens[1]]] for givens in lda_model.show_topics(formatted=False, num_topics=NUM_TOPICS, num_words=poi_count)]

cbg_topic_probabilities = {}
for cbg in cbgs_to_topics:
    current_probabilities = []
    count = 0
    for tup in cbgs_to_topics[cbg]:
        while count != tup[0]:
            current_probabilities.append(0)
            count += 1
        current_probabilities.append(tup[1])
        count += 1
    while count != NUM_TOPICS:
        current_probabilities.append(0)
        count += 1
    cbg_topic_probabilities[cbg] = current_probabilities

print_elapsed_time()
print('Reading population data...')

census_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b00.csv'), error_bad_lines=False)
cbgs_to_populations = {}
special_dict = {}  # for cbgs with no population given by the census
total_population = 0
for idx, row in census_data.iterrows():
    check_str = str(int(row['census_block_group'])).zfill(12)
    if check_str in cbg_id_set:
        if str(row['B00001e1']) in {'nan', '0.0'}:
            cbg_id_set.remove(check_str)
            cbg_ids.remove(check_str)
            continue
        cbg_population = int(int(row['B00001e1']) * PROPORTION_OF_POPULATION)
        total_population += cbg_population
        cbgs_to_populations[check_str] = cbg_population

print_elapsed_time()
print('Reading social distancing data...')

social_distancing_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_social_distancing_metrics', WEEK[:4], WEEK[5:7], WEEK[8:], '{}-social-distancing.csv'.format(WEEK)), error_bad_lines=False)
cbgs_leaving_probs = {}  # probability that a member of a cbg will leave their house each tick
population_proportions = []  # proportions list of device count / cbg population
seen_cbgs = set()
for idx, row in social_distancing_data.iterrows():
    check_str = str(int(row['origin_census_block_group'])).zfill(12)
    if check_str in cbg_id_set:
        day_leaving_prob = 1 - (int(row['completely_home_device_count']) / int(row['device_count']))
        cbgs_leaving_probs[check_str] = day_leaving_prob / total_simulation_time
        seen_cbgs.add(check_str)
for cbg in cbg_id_set - seen_cbgs:
    cbg_id_set.remove(cbg)
    cbg_ids.remove(cbg)
    total_population -= cbgs_to_populations[cbg]

cbg_probabilities = []  # probability of a single cbg being selected
for cbg in cbg_ids:
    if cbg not in cbgs_to_populations:
        cbgs_to_populations[cbg] = 0
    prob = cbgs_to_populations[cbg] / total_population
    cbg_probabilities.append(prob)

print_elapsed_time()
print('Preparing simulation...')

agents = {}  # agent_id: [topic, infected_status, critical_date, home_cbg_code]
cbgs_to_agents = {cbg: set() for cbg in cbg_ids}  # cbg: {agent id, agent id, ...}
inactive_agent_ids = set()
active_agent_ids = {t: set() for t in range(total_simulation_time)}  # expiration time: {(agent id, poi id), (agent id, poi id), ...}
poi_current_visitors = {poi: set() for poi_list in lda_documents for poi in poi_list}  # poi id: {agent id, agent id, ...}

draw = list(numpy.random.choice(cbg_ids, total_population, p=cbg_probabilities))
topic_numbers = [i for i in range(NUM_TOPICS)]
for iteration, current_cbg in enumerate(draw):
    agent_id = 'agent_' + str(iteration)
    probs = numpy.array(cbg_topic_probabilities[current_cbg])
    probs /= probs.sum()
    agent_topic = numpy.random.choice(topic_numbers, 1, p=probs)[0]
    agents[agent_id] = [agent_topic, 'I' if random.random() < 0.05 else 'S', 7, current_cbg]  # topic, infection status, critical date, cbg
    cbgs_to_agents[current_cbg].add(agent_id)
    inactive_agent_ids.add(agent_id)

aux_dict = {}  # to speed up numpy.choice with many probabilities
for topic in range(len(topics_to_pois)):
    mindex = -1
    prob_sum = 0
    while prob_sum < 0.2:
        prob_sum = sum(topics_to_pois[topic][1][mindex:])
        mindex -= 1
    aux_dict[topic] = [topics_to_pois[topic][0][mindex:], topics_to_pois[topic][1][mindex:]]
    aux_dict[topic][1] = numpy.array(aux_dict[topic][1])
    aux_dict[topic][1] /= aux_dict[topic][1].sum()
    topics_to_pois[topic][0] = topics_to_pois[topic][0][:mindex] + ['aux']
    topics_to_pois[topic][1] = numpy.array(topics_to_pois[topic][1][:mindex] + [prob_sum])
    topics_to_pois[topic][1] /= topics_to_pois[topic][1].sum()

print_elapsed_time()
print('Running simulation...')

probability_of_infection = 0.005 / SIMULATION_TICKS_PER_HOUR  # from https://hzw77-demo.readthedocs.io/en/round2/simulator_modeling.html

# Infected status
# S => Susceptible: never infected
# E => Exposed: infected on a previous day, contagious
# I => Infected: infected the same day, will be contagious tomorrow
# R => Recovered: immune to the virus, no longer contagious (not implemented yet)

for day in range(SIMULATION_DAYS):
    print('Day {}:'.format(day))
    for current_time in range(total_simulation_time):
        for poi in poi_current_visitors:
            poi_agents = list(poi_current_visitors[poi])
            for agent_id in poi_agents:
                if agents[agent_id][1] == 'I':
                    for other_agent_id in poi_agents:
                        rand = random.random()
                        if agents[other_agent_id][1] == 'S' and rand < probability_of_infection:
                            agents[other_agent_id][1] = 'E'
                            rand /= probability_of_infection
                            if rand < 0.5:
                                agents[other_agent_id][2] = day + round(rand * 6 + 2)  # ensures rand is between 2 and 5
                            else:
                                agents[other_agent_id][2] = day + round(rand * 18 - 4)  # ensures rand is between 5 and 14

        for tup in active_agent_ids[current_time]:
            inactive_agent_ids.add(tup[0])
            poi_current_visitors[tup[1]].remove(tup[0])
        # del active_agent_ids[current_time]

        to_remove = set()
        for agent_id in inactive_agent_ids:
            if random.random() < cbgs_leaving_probs[agents[agent_id][3]]:
                topic = agents[agent_id][0]
                destination = numpy.random.choice(topics_to_pois[topic][0], 1, p=topics_to_pois[topic][1])[0]
                if destination == 'aux':
                    destination = numpy.random.choice(aux_dict[topic][0], 1, p=aux_dict[topic][1])[0]
                active_agent_ids[min(current_time + SIMULATION_TICKS_PER_HOUR, total_simulation_time - 1)].add((agent_id, destination))
                poi_current_visitors[destination].add(agent_id)
                to_remove.add(agent_id)
        inactive_agent_ids -= to_remove

        current_time += 1
        if not current_time % SIMULATION_TICKS_PER_HOUR:
            print('Hour {} complete.'.format(int(current_time / SIMULATION_TICKS_PER_HOUR)))
            print_elapsed_time()
    
    for current_time in active_agent_ids:
        if current_time >= total_simulation_time:
            for tup in active_agent_ids[current_time]:
                inactive_agent_ids.add(tup[0])
                poi_current_visitors[tup[1]].remove(tup[0])
        active_agent_ids[current_time] = set()

    print('Day {} complete. Infection status:'.format(day))
    for i in ['S', 'E', 'I', 'R']:
        print('{}: {}'.format(i, len([1 for tup in agents.values() if tup[1] == i])))
    print()
    
    for key in agents:
        if agents[key][1] == 'E' and agents[key][2] == day + 1:  # https://www.nature.com/articles/s41586-020-2196-x?fbclid=IwAR1voT8K7fAlVq39RPINfrT-qTc_N4XI01fRp09-agM1v5tfXrC6NOy8-0c
            agents[key][1] = 'I'
            rand = random.random()
            if rand < 0.001349898:  # normal distribution from 4 to 10 of infectiousness, centered at 7
                agents[key][2] = day + 5  # will be infectious for 4 days
            elif rand < 0.022750132:
                agents[key][2] = day + 6  # will be infectious for 5 days
            elif rand < 0.160554782:
                agents[key][2] = day + 7  # will be infectious for 6 days
            elif rand < 0.5:
                agents[key][2] = day + 8  # will be infectious for 7 days
            elif rand < 0.8413447461:
                agents[key][2] = day + 9  # will be infectious for 8 days
            elif rand < 0.9772498681:
                agents[key][2] = day + 10  # will be infectious for 9 days
            else:
                agents[key][2] = day + 11  # will be infectious for 10 days
        elif agents[key][1] == 'I' and agents[key][2] == day + 1:
            agents[key][1] = 'R'

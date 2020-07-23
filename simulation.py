# LINKS:
# https://www.who.int/news-room/commentaries/detail/transmission-of-sars-cov-2-implications-for-infection-prevention-precautions

import pandas as pd
import gensim
import json
import random
import os
import numpy
import time
import math
import itertools
from scipy.stats import gamma
import matplotlib.pyplot as plt

NUM_TOPICS = 50
MINIMUM_RAW_VISITOR_COUNT = 30
SIMULATION_DAYS = 14
SIMULATION_HOURS_PER_DAY = 16
SIMULATION_TICKS_PER_HOUR = 4
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

# Read in data from files, filtering the data that corresponds to the area of interest
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-{}.csv'.format(area, WEEK)), error_bad_lines=False)
usable_data = data[(data.raw_visitor_counts >= MINIMUM_RAW_VISITOR_COUNT) & (data.visitor_home_cbgs != '{}')]

# From usable POIs, load cbgs, CBGs are documents, POIs are words
cbgs_to_places = {}
for row in usable_data.itertuples():
    place_id = str(row.safegraph_place_id)
    cbgs = json.loads(row.visitor_home_cbgs)
    lda_words = []
    for cbg in cbgs:
        if cbg not in cbgs_to_places:
            cbgs_to_places[cbg] = []
        # SafeGraph reports visits of cbgs as 4 if the visit is between 2-4
        cbg_frequency = random.randint(2, 4) if cbgs[cbg] == 4 else cbgs[cbg]
        # Generate POIs as words and multiply by the amount that CBG visited
        cbgs_to_places[cbg].extend([place_id] * cbg_frequency)

cbg_ids = list(cbgs_to_places.keys())
cbg_id_set = set(cbg_ids)
lda_documents = list(cbgs_to_places.values())
poi_set = {poi for poi_list in lda_documents for poi in poi_list}
poi_count = len(poi_set)

lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents) # generate documents
lda_corpus = [lda_dictionary.doc2bow(cbg) for cbg in lda_documents] # generate words
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

census_age_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b01.csv'), error_bad_lines=False)
cbgs_to_age_distribution = {} # {cbg: [% 18-49, % 50-64, % 65+] ...}
code_18_to_49 = ['B01001e7', 'B01001e8', 'B01001e9', 'B01001e10', 'B01001e11', 'B01001e12', 'B01001e13', 'B01001e14', 'B01001e15', 'B01001e31', 'B01001e32', 'B01001e33', 'B01001e34', 'B01001e35', 'B01001e36', 'B01001e37', 'B01001e38', 'B01001e39']
code_50_to_64 = ['B01001e16', 'B01001e17', 'B01001e18', 'B01001e19', 'B01001e40', 'B01001e41', 'B01001e42', 'B01001e43']
code_65_plus = ['B01001e20', 'B01001e21', 'B01001e22', 'B01001e23', 'B01001e24', 'B01001e25', 'B01001e44', 'B01001e45', 'B01001e46', 'B01001e47', 'B01001e48', 'B01001e49']
for idx, row in census_age_data.iterrows():
    check_str = str(int(row['census_block_group'])).zfill(12)
    if check_str in cbg_id_set:
        pop_18_49 = sum([int(row[a]) for a in code_18_to_49])
        pop_50_64 = sum([int(row[a]) for a in code_50_to_64])
        pop_65_over = sum([int(row[a]) for a in code_65_plus])
        pop_distribution = numpy.array([pop_18_49, pop_50_64, pop_65_over], dtype='f')
        total_pop_18_over = pop_18_49 + pop_50_64 + pop_65_over
        pop_distribution /= total_pop_18_over
        cbgs_to_age_distribution[check_str] = pop_distribution

# print(dict(itertools.islice(cbgs_to_age_distribution.items(), 10)))


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

# SETUP SIMULATION
# Generate agents for CBGs, create dictionary of active agents for each hour
# Create POI dictionary to store agents
# Randomly select a list of CBGs of the population size based on their respective probability to the population size
# Within that list of CBGs, iterate for each CBG
# Generate a randomly chosen topic based on the CBG percentage in each topic
# Once a topic and cbg are chosen, use a 5% probability to decide whether or not that agent is infected

agents = {}  # agent_id: [topic, infected_status, critical_date, home_cbg_code]
cbgs_to_agents = {cbg: set() for cbg in cbg_ids}  # cbg: {agent id, agent id, ...}
inactive_agent_ids = set()
quarantined_agent_ids = set()
active_agent_ids = {t: set() for t in range(total_simulation_time)}  # expiration time: {(agent id, poi id), (agent id, poi id), ...}
poi_current_visitors = {poi: set() for poi_list in lda_documents for poi in poi_list}  # poi id: {agent id, agent id, ...}

# https://www.nature.com/articles/s41591-020-0962-9
# https://www.acpjournals.org/doi/10.7326/M20-0504
# https://www.who.int/news-room/commentaries/detail/transmission-of-sars-cov-2-implications-for-infection-prevention-precautions
distribution_of_exposure = gamma(4, 0, 0.75) # k=4 μ=3

# temporary, normal distribution 4-10 days
distribution_of_preclinical = gamma(4, 0, 0.525)  # k=4 μ=2.1
distribution_of_clinical = gamma(4, 0, 0.725)  # k=4 μ=2.9
distribution_of_subclinical = gamma(4, 0, 1.25)  # k=4 μ=5

draw = list(numpy.random.choice(cbg_ids, total_population, p=cbg_probabilities))
topic_numbers = [i for i in range(NUM_TOPICS)]
for iteration, current_cbg in enumerate(draw):
    agent_id = 'agent_' + str(iteration)
    probs = numpy.array(cbg_topic_probabilities[current_cbg])
    probs /= probs.sum()
    agent_topic = numpy.random.choice(topic_numbers, 1, p=probs)[0]
    rand_age = numpy.random.choice(['Y', 'M', 'O'], p=cbgs_to_age_distribution[current_cbg])
    rand_infected = 0
    agent_status = 'S'
    rand = random.random()
    if rand < 0.05:
        rand /= 0.05
        if rand < 0.4:
            agent_status = 'Ia'
            rand_infected = round(distribution_of_subclinical.rvs(1)[0])
        else:
            agent_status = 'Ip'
            rand_infected = round(distribution_of_preclinical.rvs(1)[0])
    agents[agent_id] = [agent_topic, agent_status, rand_infected, current_cbg, rand_age]  # topic, infection status, critical date, cbg, age
    cbgs_to_agents[current_cbg].add(agent_id)
    inactive_agent_ids.add(agent_id)

aux_dict = {}  # to speed up numpy.choice with many probabilities {topic: list of cbgs} {cbg: poi_ids, probabilities}
for topic in range(len(topics_to_pois)):
    mindex = -1
    prob_sum = 0
    while prob_sum < 0.2:
        prob_sum = sum(topics_to_pois[topic][1][mindex:])
        mindex -= 1
    aux_dict[topic] = [topics_to_pois[topic][0][mindex:], topics_to_pois[topic][1][mindex:]] # store poi ids, probabilities
    aux_dict[topic][1] = numpy.array(aux_dict[topic][1])
    aux_dict[topic][1] /= aux_dict[topic][1].sum()
    # update old topics_to_pois with addition of aux
    topics_to_pois[topic][0] = topics_to_pois[topic][0][:mindex] + ['aux']
    topics_to_pois[topic][1] = numpy.array(topics_to_pois[topic][1][:mindex] + [prob_sum])
    topics_to_pois[topic][1] /= topics_to_pois[topic][1].sum()

print_elapsed_time()
print('Running simulation...')

# probability_of_infection = 0.005 / SIMULATION_TICKS_PER_HOUR  # from https://hzw77-demo.readthedocs.io/en/round2/simulator_modeling.html
age_susc = {'Y': 0.025, 'M': 0.045, 'O': 0.085}


def infect(p, d, i):  # poi agents, day, infectioness
    global agents
    for other_agent_id in p:
        rand = random.random()
        probability_of_infection = age_susc[agents[other_agent_id][4]] / SIMULATION_TICKS_PER_HOUR
        if agents[other_agent_id][1] == 'S' and rand < probability_of_infection * i:
            agents[other_agent_id][1] = 'E'
        rand = round(distribution_of_exposure.rvs(1)[0])
        agents[other_agent_id][2] = d + rand


def remove_expired_agents(t): # time
    global inactive_agent_ids, poi_current_visitors
    for tup in active_agent_ids[t]:
        inactive_agent_ids.add(tup[0])
        poi_current_visitors[tup[1]].remove(tup[0])
    active_agent_ids[t] = set()


def select_active_agents(t): # time
    global inactive_agent_ids, active_agent_ids, poi_current_visitors
    to_remove = set()
    for agent_id in inactive_agent_ids:
        if random.random() < cbgs_leaving_probs[agents[agent_id][3]]:
            destination_end_time = t + SIMULATION_TICKS_PER_HOUR  # improve later with a distribution
            if destination_end_time >= total_simulation_time:
                continue
            topic = agents[agent_id][0]
            destination = numpy.random.choice(topics_to_pois[topic][0], 1, p=topics_to_pois[topic][1])[0]
            if destination == 'aux':
                destination = numpy.random.choice(aux_dict[topic][0], 1, p=aux_dict[topic][1])[0]
            active_agent_ids[destination_end_time].add(
                (agent_id, destination))
            poi_current_visitors[destination].add(agent_id)
            to_remove.add(agent_id)
    inactive_agent_ids -= to_remove


def reset_all_agents():
    global inactive_agent_ids, active_agent_ids, poi_current_visitors
    for t in active_agent_ids:
        for tup in active_agent_ids[t]:
            inactive_agent_ids.add(tup[0])
            poi_current_visitors[tup[1]].remove(tup[0])
        active_agent_ids[t] = set()


def set_agent_flags(d):  # day
    global agents
    for key in agents:
        if agents[key][1] == 'E' and agents[key][2] == d + 1:  # https://www.nature.com/articles/s41586-020-2196-x?fbclid=IwAR1voT8K7fAlVq39RPINfrT-qTc_N4XI01fRp09-agM1v5tfXrC6NOy8-0c
            rand_s = random.random()
            rand = 0
            if rand_s < 0.4:
                agents[key][1] = 'Ia'
                rand = round(distribution_of_subclinical.rvs(1)[0])
            else:
                agents[key][1] = 'Ip'
                rand = round(distribution_of_preclinical.rvs(1)[0])
            agents[key][2] = d + rand
        elif agents[key][1] == 'Ip' and agents[key][2] == d+1:
            agents[key][1] = 'Ic'
            agents[key][2] = round(distribution_of_clinical.rvs(1)[0])
        elif (agents[key][1] == 'Ia' or agents[key][1] == 'Ic') and agents[key][2] == d + 1:
            agents[key][1] = 'R'


def report_status():
    stat = {'S': 0, 'E': 0, 'Ia': 0, 'Ip': 0, 'Ic': 0, 'R': 0, 'Q': 0}
    for tup in agents.values():
        stat[tup[1]] += 1
    print('Susceptible: {}'.format(stat['S']))
    print('Exposed: {}'.format(stat['E']))
    print('Infected: {}'.format(stat['Ia']+stat['Ip']+stat['Ic']))
    print('Recovered: {}'.format(stat['R']))
    print('Quarantined: {}'.format(len(quarantined_agent_ids)))


def quarantine(ag, days, d): # (agents, # days in quarantine, day) no leaving the house, contact tracing? (not implemented)
    global quarantined_agent_ids, inactive_agent_ids
    for a in ag:
        quarantined_agent_ids.add((a, d + days))
        inactive_agent_ids.remove(a)
    qq = [q for q in quarantined_agent_ids]
    for q in qq:
        if q[1] == d:
            quarantined_agent_ids.remove(q)
            inactive_agent_ids.add(q[0])


def contact_tracing(a): # no contact with anyone, used for those infected?
    # future implementation?
    return

def covid_test(a, p, ft): # agents, percent who take tests, feedback time
    # future implementation
    return

# Infected status
# S => Susceptible: never infected
# E => Exposed: infected the same day, will be contagious after incubation period
# I => Infected: infected on a previous day, contagious, 40% asymptomatic
#     Ip -> Preclinical, before clinical, not symptomatic, contagious
#     Ic -> Clinical, full symptoms, contagious
#     Ia -> Asymptomatic, 75% relative infectioness (50%)?, never show symptoms
# R => Recovered: immune to the virus, no longer contagious

for day in range(SIMULATION_DAYS):
    print('Day {}:'.format(day))
    quarantined = set()
    for current_time in range(total_simulation_time):
        remove_expired_agents(current_time)
        if current_time != total_simulation_time - 1:
            for poi in poi_current_visitors:
                poi_agents = list(poi_current_visitors[poi])
                for agent_id in poi_agents:
                    if agents[agent_id][1] == 'Ia' or agents[agent_id][1] == 'Ip':
                        infect(poi_agents, day, 0.5) # the infectioness can change, only an estimate
                    elif agents[agent_id][1] == 'Ic':
                        infect(poi_agents, day, 1)
                        quarantined.add(agent_id)
            select_active_agents(current_time)

        if not current_time % SIMULATION_TICKS_PER_HOUR:
            print('Hour {} complete.'.format(int(current_time / SIMULATION_TICKS_PER_HOUR)))
            print_elapsed_time()

    reset_all_agents()
    # quarantine(quarantined, 7, day)
    print('Day {} complete. Infection status:'.format(day))
    report_status()
    print()

    set_agent_flags(day)

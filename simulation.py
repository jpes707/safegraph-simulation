import pandas as pd
import gensim
import json
import random
import os
import numpy
import time
import math
import itertools
import pickle
import numpy as np
import statistics
import gzip
import queue
from distfit import distfit
import scipy.stats

NUM_TOPICS = 50  # number of LDA topics
SIMULATION_DAYS = 365 * 5  # number of "days" the simulation runs for
SIMULATION_TICKS_PER_HOUR = 4  # integer, number of ticks per simulation "hour"
PROPORTION_OF_POPULATION = 1  # 0.2 => 20% of the actual population is simulated
PROPORTION_INITIALLY_INFECTED = 0.001  # 0.05 => 5% of the simulated population is initially infected or exposed
PROPENSITY_TO_LEAVE = 1  # 0.2 => people are only 20% as likely to leave the house as compared to normal
NUMBER_OF_DWELL_SAMPLES = 5  # a higher number decreases POI dwell time variation and allows less outliers
MAX_DWELL_TIME = 16  # maximum dwell time at any POI (hours)
QUARANTINE_DURATION = 10  # number of days a quarantine lasts after an agent begins to show symptoms
MAXIMUM_INTERACTIONS_PER_TICK = 5  # integer, maximum number of interactions an infected person can have with others per tick
ALPHA = 0  # 0.4 => 40% of the population is quarantined in their house for the duration of the simulation
SYMPTOMATIC_QUARANTINES = False
WEAR_MASKS = False
SOCIAL_DISTANCING = False
EYE_PROTECTION = False
CLOSED_POI_TYPES = {  # closed POI types (from SafeGraph Core Places "sub_category")
}

start_time = time.time()
daily_simulation_time = SIMULATION_TICKS_PER_HOUR * 24
total_simulation_time = SIMULATION_DAYS * daily_simulation_time


def adj_sig_figs(a, n=3):  # truncate the number a to have n significant figures
    if not a:
        return 0
    dec = int(math.log10(abs(a)) // 1)
    z = int((a * 10 ** (n - 1 - dec) + 0.5) // 1) / 10 ** (n - 1 - dec)
    return str(z) if z % 1 else str(int(z))


def print_elapsed_time():
    print('Total time elapsed: {}s'.format(adj_sig_figs(time.time() - start_time, 5)))


# prompt user whether or not to used cached data in order to reduce initial simulation load time
raw_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_cache.p')
agents_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents_cache.p')
agents_loaded = False
use_raw_cache = input('Use file data cache ([y]/n)? ')
if use_raw_cache == '' or use_raw_cache == 'y':  # obtains cached variables from the file data cache
    raw_cache_file = open(raw_cache_path, 'rb')
    (cbg_ids, lda_documents, cbgs_to_households, cbg_topic_probabilities, topics_to_pois, cbgs_leaving_probs, dwell_distributions, poi_type, topic_hour_distributions, topics_to_pois_by_hour, under_16_chance) = pickle.load(raw_cache_file)
    use_agents_cache = input('Use agents cache ([y]/n)? ')  # obtains cached variables from agent file data cache
    if use_agents_cache == '' or use_agents_cache == 'y':
        agents_cache_file = open(agents_cache_path, 'rb')
        (agents, households, cbgs_to_agents, inactive_agent_ids, active_agent_ids, poi_current_visitors, Ipa_queue_tups, Ic_queue_tups, R_queue_tups, quarantine_queue_tups) = pickle.load(agents_cache_file)
        agents_loaded = True
else:  # loads and caches data from files depending on user input
    # prompts for user input
    STATE_ABBR = input('State abbreviation (default: VA): ')
    if STATE_ABBR == '':
        STATE_ABBR = 'VA'
    LOCALITY = input('Locality (default: Fairfax): ')  # can be a county, borough, or parish
    if LOCALITY == '':
        LOCALITY = 'Fairfax'
    WEEK = input('Week (default: 2019-10-28): ')
    if WEEK == '':
        WEEK = '2019-10-28'

    # checks locality type
    if STATE_ABBR == 'AK':
        LOCALITY_TYPE = 'borough'
    elif STATE_ABBR == 'LA':
        LOCALITY_TYPE = 'parish'
    else:
        LOCALITY_TYPE = 'county'

    locality_name = (LOCALITY + ' ' + LOCALITY_TYPE).title()  # e.g. "Fairfax County"
    full_name = '{}, {}'.format(locality_name, STATE_ABBR.upper())  # e.g. "Fairfax County, VA"
    area = locality_name.lower().replace(' ', '-') + '-' + STATE_ABBR.lower()  # e.g. "fairfax-county-va"

    print('Reading POI data...')

    # read in data from weekly movement pattern files, filtering the data that corresponds to the area of interest
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(WEEK), '{}-{}.csv'.format(area, WEEK)), error_bad_lines=False)
    usable_data = data[(data.visitor_home_cbgs != '{}')]

    # from usable POIs, load cbgs, CBGs are documents, POIs are words
    cbgs_to_pois = {}
    dwell_distributions = {}
    cbgs_to_places = {}
    poi_set = set()
    poi_type = {}
    poi_hour_counts = {}  # poi id: length 24 list of poi visits by hour
    for row in usable_data.itertuples():
        place_id = str(row.safegraph_place_id)
        poi_set.add(place_id)
        dwell_distributions[place_id] = eval(row.dwell_distribution)
        place_type = str(row.sub_category)
        if place_type not in poi_type:
            poi_type[place_type] = {place_id}
        else:
            poi_type[place_type].add(place_id)
        weekly_hours_info = json.loads(row.visits_by_each_hour)
        hourly_poi = [0] * 24
        for idx, num in enumerate(weekly_hours_info):
            hourly_poi[idx % 24] += num
        poi_hour_counts[place_id] = hourly_poi
        cbgs = json.loads(row.visitor_home_cbgs)
        lda_words = []
        for cbg in cbgs:
            if not cbg.startswith('51059'):  # excludes all POIs outside of Fairfax County, must be removed later!
                continue
            if cbg not in cbgs_to_pois:
                cbgs_to_pois[cbg] = []
            cbg_frequency = random.randint(2, 4) if cbgs[cbg] == 4 else cbgs[cbg]  # SafeGraph reports POI CBG frequency as 4 if the visit count is between 2-4
            cbgs_to_pois[cbg].extend([place_id] * cbg_frequency)  # generate POIs as "words" and multiply by the amount that CBG visited
    
    print_elapsed_time()
    print('Running LDA and creating initial distributions...')

    cbg_ids = list(cbgs_to_pois.keys())
    cbg_id_set = set(cbg_ids)
    lda_documents = list(cbgs_to_pois.values())
    poi_set = {poi for poi_list in lda_documents for poi in poi_list}
    poi_count = len(poi_set)

    lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents)  # generate "documents" for gensim
    lda_corpus = [lda_dictionary.doc2bow(cbg) for cbg in lda_documents]  # generate "words" for gensim
    cbg_to_bow = dict(zip(cbg_ids, lda_corpus))
    lda_model = gensim.models.LdaModel(lda_corpus, num_topics=NUM_TOPICS, id2word=lda_dictionary)

    cbgs_to_topics = dict(zip(cbg_ids, list(lda_model.get_document_topics(lda_corpus, minimum_probability=0))))  # {'510594301011': [(0, 5.5570694e-05), (1, 5.5570694e-05), (2, 5.5570694e-05), (3, 5.5570694e-05), ...], ...}
    topics_to_pois = [[[tup[0] for tup in givens[1]], [tup[1] for tup in givens[1]]] for givens in lda_model.show_topics(formatted=False, num_topics=NUM_TOPICS, num_words=poi_count)]  # [[poi id list, probability dist], [poi id list, probability dist], ...]
    
    topic_hour_distributions = []  # topic: [0.014925373134328358, 0.0, 0.014925373134328358, 0.014925373134328358, 0.014925373134328358, 0.014925373134328358, 0.014925373134328358, 0.08955223880597014, 0.029850746268656716, 0.0, 0.04477611940298507, 0.029850746268656716, 0.029850746268656716, 0.04477611940298507, 0.11940298507462686, 0.04477611940298507, 0.05970149253731343, 0.1044776119402985, 0.04477611940298507, 0.05970149253731343, 0.07462686567164178, 0.08955223880597014, 0.014925373134328358, 0.029850746268656716] (chance of leaving by hour, adds up to 1)
    topics_to_pois_by_hour = []  # topic: [hour: poi info lists, hour: poi info lists, ...] (topics_to_pois but weighted per hour)
    for topic, givens in enumerate(topics_to_pois):  # iterate through each topic to establish chances of leaving per hour
        current_hours_dist = numpy.array([0.0] * 24)
        poi_ids = givens[0]
        poi_probabilities = givens[1]
        current_pois_by_hour = []
        for hour in range(24):
            hourly_sum = 0
            hourly_poi_ids = []
            hourly_poi_probabilities = []
            for idx, poi_id in enumerate(poi_ids):
                if poi_hour_counts[poi_ids[idx]][hour]:
                    hourly_sum += poi_probabilities[idx] * poi_hour_counts[poi_id][hour]
                    hourly_poi_ids.append(poi_id)
                    hourly_poi_probabilities.append(poi_probabilities[idx] * poi_hour_counts[poi_id][hour] / sum(poi_hour_counts[poi_id]))
            hourly_poi_probabilities_sum = sum(hourly_poi_probabilities)
            for idx, prob in enumerate(hourly_poi_probabilities):
                hourly_poi_probabilities[idx] = prob / hourly_poi_probabilities_sum
            current_pois_by_hour.append([hourly_poi_ids, hourly_poi_probabilities])
            current_hours_dist[hour] = hourly_sum
        current_hours_dist /= current_hours_dist.sum()
        topic_hour_distributions.append(current_hours_dist)
        topics_to_pois_by_hour.append(current_pois_by_hour)

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
    print('Reading CBG population data...')

    census_population_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b01.csv'), error_bad_lines=False)
    cbgs_to_real_populations = {}
    total_real_population = 0
    for idx, row in census_population_data.iterrows():
        check_str = str(int(row['census_block_group'])).zfill(12)
        if check_str in cbg_id_set:
            if str(row['B01001e1']) in {'nan', '0.0'}:
                cbg_id_set.remove(check_str)
                cbg_ids.remove(check_str)
            else:
                cbg_real_population = int(int(row['B01001e1']) * PROPORTION_OF_POPULATION)
                total_real_population += cbg_real_population
                cbgs_to_real_populations[check_str] = cbg_real_population

    print_elapsed_time()
    print('Reading household data...')

    census_household_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b11.csv'), error_bad_lines=False)
    cbgs_to_households = {}  # {cbg: [# nonfamily households of size 1, # nonfamily households of size 2, ... # nonfamily households of size 7+, # family households of size 2, # family households of size 3, ... # family households of size 7+]}
    cbgs_to_populations = {}
    total_population = 0
    for _, row in census_household_data.iterrows():
        check_str = str(int(row['census_block_group'])).zfill(12)
        if check_str in cbg_id_set:
            arr = []
            for ext in range(10, 17):  # appends nonfamily household counts
                arr.append(int(int(row['B11016e{}'.format(ext)]) * PROPORTION_OF_POPULATION))
            for ext in range(3, 9):  # appends family household counts
                arr.append(int(int(row['B11016e{}'.format(ext)]) * PROPORTION_OF_POPULATION))
            if arr == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                cbg_id_set.remove(check_str)
                cbg_ids.remove(check_str)
            else:
                household_based_cbg_population = sum([(idx % 7 + idx // 7 + 1) * arr[idx] for idx in range(13)])
                underestimate_corrector = cbgs_to_real_populations[check_str] / household_based_cbg_population
                arr = [int(elem * underestimate_corrector) for elem in arr]
                cbgs_to_households[check_str] = arr
                cbg_population = sum([(idx % 7 + idx // 7 + 1) * arr[idx] for idx in range(13)])
                total_population += cbg_population
                cbgs_to_populations[check_str] = cbg_population
    print(r'The simulated population will be {}% of the real population.'.format(adj_sig_figs(100 * total_population / total_real_population)))

    print_elapsed_time()
    print('Reading age data...')

    census_age_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b01.csv'), error_bad_lines=False)
    codes_under_15 = ['B01001e3', 'B01001e4', 'B01001e5', 'B01001e27', 'B01001e28', 'B01001e29']
    codes_15_to_17 = ['B01001e6', 'B01001e30']
    under_16_chance = {}  # {cbg: chance that a non-householder member of a family household is under 16, ...} (assumes children under 16 cannot live in nonfamily households and that children 16 and older can be householders due to emancipation)
    for _, row in census_age_data.iterrows():
        check_str = str(int(row['census_block_group'])).zfill(12)
        if check_str in cbg_id_set:
            pop_under_15 = sum([int(row[elem]) for elem in codes_under_15])
            pop_15_to_17 = sum([int(row[elem]) for elem in codes_15_to_17])
            pop_under_16 = pop_under_15 + pop_15_to_17 / 3  # assumes 1/3 of the population from age 15 to 17 in the CBG is 15
            family_households_non_householder_population = sum([(idx % 7 + idx // 7 + 1) * cbgs_to_households[check_str][idx] for idx in range(7, 13)]) - sum([cbgs_to_households[check_str][idx] for idx in range(7, 13)])  # number of family household members - number of family householders, see household generation code below for sense of the variable values
            if family_households_non_householder_population:
                under_16_chance[check_str] = pop_under_16 / family_households_non_householder_population
            else:
                under_16_chance[check_str] = 0

    print_elapsed_time()
    print('Reading social distancing data...')
    
    cbg_device_counts = {}  # cbg: [completely_home_device_count, device_count]
    total_devices_tracked = 0
    for date in pd.date_range(WEEK, periods=7, freq='D'):
        print(date)
        social_distancing_data = pd.read_csv(gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_social_distancing_metrics', str(date.year), str(date.month).zfill(2), str(date.day).zfill(2), '{}-{}-{}-social-distancing.csv.gz'.format(str(date.year), str(date.month).zfill(2), str(date.day).zfill(2))), 'rb'), error_bad_lines=False)
        for idx, row in social_distancing_data.iterrows():
            check_str = str(int(row['origin_census_block_group'])).zfill(12)
            if check_str in cbg_id_set:
                if check_str not in cbg_device_counts:
                    cbg_device_counts[check_str] = [0, 0]
                cbg_device_counts[check_str][0] += int(row['completely_home_device_count'])
                cbg_device_counts[check_str][1] += int(row['device_count'])
                total_devices_tracked += cbg_device_counts[check_str][1]
    for cbg in cbg_id_set - set(cbg_device_counts.keys()):  # removes cbg due to lack of data
        cbg_id_set.remove(cbg)
        cbg_ids.remove(cbg)
        total_population -= cbgs_to_populations[cbg]
    cbgs_leaving_probs = {}  # probability that a member of a cbg will leave their house each tick
    for cbg in cbg_device_counts:
        cbgs_leaving_probs[cbg] = 1 - (cbg_device_counts[cbg][0] / cbg_device_counts[cbg][1])
    print('Average percentage of population tracked per day: {}%'.format(adj_sig_figs(100 * total_devices_tracked / (total_population * 7))))

    print_elapsed_time()
    print('Caching raw data...')

    raw_cache_data = (cbg_ids, lda_documents, cbgs_to_households, cbg_topic_probabilities, topics_to_pois, cbgs_leaving_probs, dwell_distributions, poi_type, topic_hour_distributions, topics_to_pois_by_hour, under_16_chance)
    raw_cache_file = open(raw_cache_path, 'wb')
    pickle.dump(raw_cache_data, raw_cache_file)

    print_elapsed_time()

raw_cache_file.close()

# https://www.nature.com/articles/s41591-020-0962-9
# https://www.acpjournals.org/doi/10.7326/M20-0504
# https://www.who.int/news-room/commentaries/detail/transmission-of-sars-cov-2-implications-for-infection-prevention-precautions
distribution_of_exposure = scipy.stats.gamma(4, 0, 0.75) # k=4 μ=3 => midpoint is 2.754 days

# temporary, normal distribution 4-10 days
distribution_of_preclinical = scipy.stats.gamma(4, 0, 0.525)  # k=4 μ=2.1 => midpoint is 1.928 days
distribution_of_clinical = scipy.stats.gamma(4, 0, 0.725)  # k=4 μ=2.9 => midpoint is 2.662 days
distribution_of_subclinical = scipy.stats.gamma(4, 0, 1.25)  # k=4 μ=5 => midpoint is 4.590 days

topic_numbers = [i for i in range(NUM_TOPICS)]

if not agents_loaded:
    print('Preparing simulation... 0%')

    # SETUP SIMULATION
    # Generate agents for CBGs, create dictionary of active agents for each hour
    # Create POI dictionary to store agents
    # Randomly select a list of CBGs of the population size based on their respective probability to the population size
    # Within that list of CBGs, iterate for each CBG
    # Generate a randomly chosen topic based on the CBG percentage in each topic
    # Once a topic and cbg are chosen, use a 0.1% probability to decide whether or not that agent is infected

    agents = {}  # agent_id: [topic, infected_status, (current poi, expiration time) or None, home_cbg_code]
    households = {}  # household_id: {agent_id, agent_id, ...}
    cbgs_to_agents = {cbg: set() for cbg in cbg_ids}  # cbg: {agent_id, agent_id, ...}
    inactive_agent_ids = set()  # {agent_id, agent_id, ...}
    active_agent_ids = {t: set() for t in range(total_simulation_time)}  # expiration_time: {(agent_id, poi_id), (agent_id, poi_id), ...}
    poi_current_visitors = {poi: set() for poi_list in lda_documents for poi in poi_list}  # poi id: {agent_id, agent_id, ...}
    Ipa_queue_tups = []
    Ic_queue_tups = []
    R_queue_tups = []
    quarantine_queue_tups = []

    agent_count = 0
    household_count = 0


    def add_agent(current_cbg, household_id, probs, possibly_child):
        global agent_count
        agent_id = 'agent_{}'.format(agent_count)
        agent_count += 1
        agent_topic = numpy.random.choice(topic_numbers, 1, p=probs)[0]
        agent_status = 'S'
        if possibly_child:
            permanently_quarantined = random.random() < under_16_chance[current_cbg]
        else:
            permanently_quarantined = random.random() < ALPHA  # prohibits agent from ever leaving their house and ensures they are not initially infected if True
        parameter_2 = None
        if not permanently_quarantined:
            rand = random.random()
            if rand < PROPORTION_INITIALLY_INFECTED:
                rand /= PROPORTION_INITIALLY_INFECTED
                if rand < 0.23076923076:  # 2.754 / (2.754 + 1.928 + 2.662 + 4.590), from exposure gamma distributions
                    agent_status = 'E'
                    Ipa_queue_tups.append((round(distribution_of_exposure.rvs(1)[0] * daily_simulation_time), agent_id))
                    inactive_agent_ids.add(agent_id)
                elif rand < 0.53846153845:  # 40% of the remaining value, represents asymptomatic (subclinical) cases
                    agent_status = 'Ia'
                    R_queue_tups.append((round(distribution_of_subclinical.rvs(1)[0] * daily_simulation_time), agent_id))
                    inactive_agent_ids.add(agent_id)
                elif rand < 0.7323278029:  # 1.928 / 4.590 of the remaining value, represents preclinical cases
                    agent_status = 'Ip'
                    Ic_queue_tups.append((round(distribution_of_preclinical.rvs(1)[0] * daily_simulation_time), agent_id))
                    inactive_agent_ids.add(agent_id)
                else:  # represents symptomatic (clinical) cases
                    agent_status = 'Ic'
                    R_queue_tups.append((round(distribution_of_clinical.rvs(1)[0] * daily_simulation_time), agent_id))
                    if SYMPTOMATIC_QUARANTINES:
                        quarantine_queue_tups.append((QUARANTINE_DURATION * 24 * SIMULATION_TICKS_PER_HOUR, agent_id))
                        parameter_2 = 'quarantined'
                    else:
                        inactive_agent_ids.add(agent_id)
            else:
                inactive_agent_ids.add(agent_id)
        else:
            inactive_agent_ids.add(agent_id)
        agents[agent_id] = [agent_topic, agent_status, parameter_2, current_cbg]
        cbgs_to_agents[current_cbg].add(agent_id)
        households[household_id].add(agent_id)


    benchmark = 5
    for i, current_cbg in enumerate(cbg_ids):
        probs = numpy.array(cbg_topic_probabilities[current_cbg])
        probs /= probs.sum()
        for idx, cbg_household_count in enumerate(cbgs_to_households[current_cbg]):
            current_household_size = idx % 7 + idx // 7 + 1  # will end up being one more than this for family households
            for _ in range(cbg_household_count):
                household_id = 'household_{}'.format(household_count)
                household_count += 1
                households[household_id] = set()
                if idx >= 7:  # family households
                    add_agent(current_cbg, household_id, probs, False)  # householder, cannot be a child
                    for _ in range(1, current_household_size):  # adds additional family members, can be children
                        add_agent(current_cbg, household_id, probs, True)
                else:
                    for _ in range(current_household_size):
                        add_agent(current_cbg, household_id, probs, False)
        completion = i / len(cbg_ids) * 100
        if completion >= benchmark:
            print('Preparing simulation... {}%'.format(adj_sig_figs(benchmark)))
            benchmark += 5

    print_elapsed_time()
    print('Caching agents data...')

    agents_cache_data = (agents, households, cbgs_to_agents, inactive_agent_ids, active_agent_ids, poi_current_visitors, Ipa_queue_tups, Ic_queue_tups, R_queue_tups, quarantine_queue_tups)
    agents_cache_file = open(agents_cache_path, 'wb')
    pickle.dump(agents_cache_data, agents_cache_file)

    print_elapsed_time()

agents_cache_file.close()
print('Number of agents: {}'.format(len(agents)))

print('Normalizing probabilities...')

aux_dict = {}  # an auxiliary POI dictionary for the least likely POIs per topic to speed up numpy.choice with many probabilities {topic: list of cbgs} {cbg: poi_ids, probabilities}
for topic in range(len(topics_to_pois_by_hour)):  # iterates through each topic
    aux_dict[topic] = [[] for i in range(24)]
    for t in range(24):
        minimum_list_index = -1
        prob_sum = 0
        while prob_sum < 0.2:  # the probability of selecting any POI in aux_dict is 20%
            prob_sum = sum(topics_to_pois_by_hour[topic][t][1][minimum_list_index:])
            minimum_list_index -= 1
        aux_dict[topic][t] = [topics_to_pois_by_hour[topic][t][0][minimum_list_index:], topics_to_pois_by_hour[topic][t][1][minimum_list_index:]]  # [poi ids list, poi probabilities list for numpy.choice]
        aux_dict[topic][t][1] = numpy.array(aux_dict[topic][t][1])  # converts poi probabilities list at aux_dict[topic][t][1] from a list to a numpy array for numpy.choice
        aux_dict[topic][t][1] /= aux_dict[topic][t][1].sum()  # ensures the sum of aux_dict[topic][t][1] is 1 for numpy.choice
        # update old topics_to_pois with addition of aux, representing the selection of a POI from aux_dict
        topics_to_pois_by_hour[topic][t][0] = topics_to_pois_by_hour[topic][t][0][:minimum_list_index] + ['aux']
        topics_to_pois_by_hour[topic][t][1] = numpy.array(topics_to_pois_by_hour[topic][t][1][:minimum_list_index] + [prob_sum])
        topics_to_pois_by_hour[topic][t][1] /= topics_to_pois_by_hour[topic][t][1].sum()

print_elapsed_time()
print('Creating queues...')

# Infected status
# S => Susceptible: never infected
# E => Exposed: infected the same day, will be contagious after incubation period
# Infected: 60% of cases are Ip (presymptomatic) then Ic (symptomatic), 40% of cases are Ia (asymptomatic, these agents never show symptoms)
#     Ip => Preclinical, before clinical, not symptomatic, contagious
#     Ic => Clinical, full symptoms, contagious
#     Ia => Asymptomatic (subclinical), 75% relative infectioness, never show symptoms
# R => Recovered: immune to the virus, no longer contagious

Ipa_queue = queue.PriorityQueue()  # queue to transition to Ip or Ia
for tup in Ipa_queue_tups:
    Ipa_queue.put(tup)
del Ipa_queue_tups
Ic_queue = queue.PriorityQueue()  # queue to transition to Ip or Ia
for tup in Ic_queue_tups:
    Ic_queue.put(tup)
del Ic_queue_tups
R_queue = queue.PriorityQueue()  # queue to transition to R
for tup in R_queue_tups:
    R_queue.put(tup)
del R_queue_tups
quarantine_queue = queue.PriorityQueue()
for tup in quarantine_queue_tups:
    quarantine_queue.put(tup)
del quarantine_queue_tups

print_elapsed_time()
print('Running simulation...')

# For COVID-19, a close contact is defined as ay individual who was within 6 feet of an infected person for at least 15 minutes starting from 2 days before illness onset (or, for asymptomatic patients, 2 days prior to positive specimen collection) until the time the patient is isolated. (https://www.cdc.gov/coronavirus/2019-ncov/php/contact-tracing/contact-tracing-plan/contact-tracing.html)
secondary_attack_rate = 0.05  # DO NOT DIVIDE BY SIMULATION_TICKS_PER_HOUR, chance of contracting the virus on close contact with someone, from https://jamanetwork.com/journals/jama/fullarticle/2768396
asymptomatic_relative_infectiousness = 0.75  # https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
mask_reduction_factor = 3.1 / 17.4  # https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)31142-9/fulltext
social_distancing_reduction_factor = 2.6 / 12.8  # https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)31142-9/fulltext
eye_protection_reduction_factor = 5.5 / 16.0 # https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)31142-9/fulltext


def get_dwell_time(dwell_tuple):  # given a cached tuple from the dwell_distributions dictionary for a specific POI, return a dwell time in ticks
    dwell_time_minutes = 0  # represents dwell time in minutes (not ticks)
    while dwell_time_minutes <= 0:
        if len(dwell_tuple) == 3:
            dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(loc=dwell_tuple[1], scale=dwell_tuple[2], size=NUMBER_OF_DWELL_SAMPLES))
        elif len(dwell_tuple) == 4:
            dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(dwell_tuple[1], loc=dwell_tuple[2], scale=dwell_tuple[3], size=NUMBER_OF_DWELL_SAMPLES))
        else:
            dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(dwell_tuple[1], dwell_tuple[2], loc=dwell_tuple[3], scale=dwell_tuple[4], size=NUMBER_OF_DWELL_SAMPLES))
    dwell_time_ticks = int(round(dwell_time_minutes * SIMULATION_TICKS_PER_HOUR / 60))  # represents dwell time in ticks
    if dwell_time_ticks == 0:
        dwell_time_ticks = 1
    elif dwell_time_ticks > MAX_DWELL_TIME * SIMULATION_TICKS_PER_HOUR:
        dwell_time_ticks = MAX_DWELL_TIME * SIMULATION_TICKS_PER_HOUR
    return dwell_time_ticks


def infect(agent_id, current_time):  # infects an agent with the virus
    global infection_counts_by_day, Ipa_queue
    if agents[agent_id][1] == 'S':
        agents[agent_id][1] = 'E'
        Ipa_queue.put((current_time + round(distribution_of_exposure.rvs(1)[0] * daily_simulation_time), agent_id))
        infection_counts_by_day[current_time // daily_simulation_time] += 1


def poi_infect(current_poi_agents, current_time, infectiousness):  # poi_agents, day, infectioness
    global agents
    for other_agent_id in current_poi_agents:
        rand = random.random()
        if rand < secondary_attack_rate * infectiousness:  # assumes direct contact with other agent and a contact limit, does not use per hour probabilities
            infect(other_agent_id, current_time)


def remove_expired_agents(t):  # removes agents from POIs whose visits end at time t
    global inactive_agent_ids, poi_current_visitors
    for tup in active_agent_ids[t]:
        inactive_agent_ids.add(tup[0])
        poi_current_visitors[tup[1]].remove(tup[0])
        agents[tup[0]][2] = None
    del active_agent_ids[t]


def select_active_agents(t):  # puts agents into POIs whose visits start at at time t
    global inactive_agent_ids, active_agent_ids, poi_current_visitors
    hour = t // SIMULATION_TICKS_PER_HOUR % 24
    to_remove = set()
    for agent_id in inactive_agent_ids:
        topic = agents[agent_id][0]
        if random.random() < cbgs_leaving_probs[agents[agent_id][3]] * topic_hour_distributions[topic][hour] / SIMULATION_TICKS_PER_HOUR:
            destination = destination = numpy.random.choice(topics_to_pois_by_hour[topic][hour][0], 1, p=topics_to_pois_by_hour[topic][hour][1])[0]
            if destination == 'aux':
                destination = numpy.random.choice(aux_dict[topic][hour][0], 1, p=aux_dict[topic][hour][1])[0]
            if destination in closed_pois:
                continue
            destination_end_time = t + get_dwell_time(dwell_distributions[destination])
            active_agent_ids[destination_end_time].add((agent_id, destination))
            poi_current_visitors[destination].add(agent_id)
            agents[agent_id][2] = (destination, destination_end_time)
            to_remove.add(agent_id)
    inactive_agent_ids -= to_remove


def infect_active_agents(current_time):
    for poi in poi_current_visitors:
        if poi not in closed_pois and len(poi_current_visitors[poi]) > 1:
            poi_agents = list(poi_current_visitors[poi])
            max_interactions = min(len(poi_agents) - 1, MAXIMUM_INTERACTIONS_PER_TICK)
            for agent_idx, agent_id in enumerate(poi_agents):
                if agents[agent_id][1] == 'Ia' or agents[agent_id][1] == 'Ip':
                    possible_agents = poi_agents[:agent_idx] + poi_agents[agent_idx+1:]
                    poi_infect(np.random.choice(possible_agents, max_interactions), current_time, asymptomatic_relative_infectiousness)
                elif agents[agent_id][1] == 'Ic':
                    possible_agents = poi_agents[:agent_idx] + poi_agents[agent_idx+1:]
                    poi_infect(np.random.choice(possible_agents, max_interactions), current_time, 1)


def household_transmission(current_time):  # simulates a day's worth of of household transmission in every household
    for household in households.values():
        base_transmission_probability = .091/4.59 if len(household) < 6 else .204/4.59  # https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30471-0/fulltext
        for agent_id in household:
            if agents[agent_id][1] == 'Ia' or agents[agent_id][1] == 'Ip':
                for other_agent_id in household:
                    if random.random() < base_transmission_probability * asymptomatic_relative_infectiousness:
                        infect(other_agent_id, current_time)
            elif agents[agent_id][1] == 'Ic':
                for other_agent_id in household:
                    if random.random() < base_transmission_probability:
                        infect(other_agent_id, current_time)


def quarantine(agent, days, current_time):  # agents, number of days in quarantine, current day => agents cannot leave house during quarantine
    global quarantine_queue, inactive_agent_ids
    quarantine_queue.put((current_time + days * daily_simulation_time, agent))
    if agents[agent][2] == 'quarantined':
        return
    elif agents[agent][2] == None:
        inactive_agent_ids.remove(agent)
    else:
        visit_info = agents[agent][2]
        poi_current_visitors[visit_info[0]].remove(agent)
        active_agent_ids[visit_info[1]].remove((agent, visit_info[0]))
    agents[agent][2] = 'quarantined'


def set_agent_flags(current_time):  # changes agent infection status
    global agents, Ipa_queue, Ic_queue, R_queue, quarantine_queue
    while Ipa_queue and Ipa_queue.queue[0][0] <= current_time + 1:  # https://www.nature.com/articles/s41586-020-2196-x?fbclid=IwAR1voT8K7fAlVq39RPINfrT-qTc_N4XI01fRp09-agM1v5tfXrC6NOy8-0c
        key = Ipa_queue.get()[1]
        rand_s = random.random()
        if rand_s < 0.4:  # subclinical (asymptomatic) cases, never show symptoms
            agents[key][1] = 'Ia'
            R_queue.put((current_time + round(distribution_of_subclinical.rvs(1)[0] * daily_simulation_time), key))
        else:  # preclinical cases, will be symptomatic in the future
            agents[key][1] = 'Ip'
            Ic_queue.put((current_time + round(distribution_of_preclinical.rvs(1)[0] * daily_simulation_time), key))
    while Ic_queue and Ic_queue.queue[0][0] <= current_time + 1:
        key = Ic_queue.get()[1]
        agents[key][1] = 'Ic'
        if SYMPTOMATIC_QUARANTINES:
            quarantine(key, QUARANTINE_DURATION, current_time)
        R_queue.put((current_time + round(distribution_of_clinical.rvs(1)[0] * daily_simulation_time), key))
    while R_queue and R_queue.queue[0][0] <= current_time + 1:
        key = R_queue.get()[1]
        agents[key][1] = 'R'
    if SYMPTOMATIC_QUARANTINES:
        while quarantine_queue and quarantine_queue.queue[0][0] <= current_time + 1:
            key = quarantine_queue.get()[1]
            inactive_agent_ids.add(key)
            agents[key][2] = None


def report_status():  # prints status
    global infection_counts_by_day
    stat = {'S': 0, 'E': 0, 'Ia': 0, 'Ip': 0, 'Ic': 0, 'R': 0, 'Q': 0}
    for tup in agents.values():
        stat[tup[1]] += 1
    print('Susceptible: {}'.format(stat['S']))
    print('Exposed: {}'.format(stat['E']))
    print('Infected: {}'.format(stat['Ia']+stat['Ip']+stat['Ic']))
    print('Recovered: {}'.format(stat['R']))
    print('New cases per day: {}'.format(list(zip([i for i in range(len(infection_counts_by_day))], infection_counts_by_day))))
    print()


def close_poi_type(current_poi_type):  # e.g. "Restaurants and Other Eating Places"
    global closed_pois
    closed_pois |= poi_type[current_poi_type]


def set_propensity_to_leave():
    for cbg in cbgs_leaving_probs:
        cbgs_leaving_probs[cbg] *= PROPENSITY_TO_LEAVE


# prescriptions
if WEAR_MASKS:
    secondary_attack_rate *= mask_reduction_factor
if SOCIAL_DISTANCING:
    secondary_attack_rate *= social_distancing_reduction_factor
if EYE_PROTECTION:
    secondary_attack_rate *= eye_protection_reduction_factor
closed_pois = set()
for current_poi_type in CLOSED_POI_TYPES:
    close_poi_type(current_poi_type)
set_propensity_to_leave()

print('Day 0:')
infection_counts_by_day = [0]
for current_time in range(total_simulation_time):
    remove_expired_agents(current_time)
    select_active_agents(current_time)
    infect_active_agents(current_time)
    set_agent_flags(current_time)
    if not (current_time + 1) % SIMULATION_TICKS_PER_HOUR:  # if the next tick marks the start of an hour
        print('Hour {} ({}) complete.'.format(int(current_time // SIMULATION_TICKS_PER_HOUR % 24), int(current_time / SIMULATION_TICKS_PER_HOUR)))
        print_elapsed_time()
        if not (current_time + 1) % daily_simulation_time:  # if the next tick marks midnight
            print(current_time)
            day = current_time // daily_simulation_time
            household_transmission(current_time)
            print('Daily household transmission complete.')
            print_elapsed_time()
            print('Day {} complete. Infection status:'.format(day))
            report_status()
            print('Day {}:'.format(day + 1))
            infection_counts_by_day.append(0)

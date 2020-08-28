from datetime import datetime
import pandas as pd
import gensim
import random
import json
import os
import sys
import numpy
import time
import math
import itertools
import statistics
import gzip
import scipy.stats
import configparser

# BEGIN DEFAULT CONFIGURATION FOR LINTER, EDITING THIS DOES NOTHING!!! USE .cfg FILES INSTEAD!!!

# preliminary parameters
LOCALITY = 'Fairfax'  # can be a county, borough, or parish; independent cities (e.g. Baltimore City) are not currently compatible
STATE_ABBR = 'VA'  # state abbreviation of where the locality is located
WEEK = '2019-10-28'  # start date of the week to base data off of, must be a Monday
NUM_TOPICS = 50  # number of Latent Dirichlet Allocation (LDA) topics
RANDOM_SEED = 1  # SETTING THIS TO 0 USES THE SYSTEM CLOCK!! random seed for all random events in this script
PROPORTION_OF_POPULATION = 1  # 0.2 => 20% of the actual population is simulated, a number lower than 1 will cause the curve to "flatten" because all POIs are still simulated
PROPENSITY_TO_LEAVE = 1  # 0.2 => people are only 20% as likely to leave the house as compared to normal

# agent generation parameters
SIMULATION_TICKS_PER_HOUR = 4  # integer, number of ticks per simulation "hour"

# runtime parameters
MAX_DWELL_TIME = 16  # maximum dwell time at any POI (hours)
NUMBER_OF_DWELL_SAMPLES = 5  # a higher number decreases POI dwell time variation and allows less outliers

# END DEFAULT CONFIGURATION FOR LINTER


def update_outfile():
    global outfile
    outfile.flush()
    os.fsync(outfile.fileno())


def adj_sig_figs(a, n=3):  # truncate the number a to have n significant figures
    if not a:
        return 0
    dec = int(math.log10(abs(a)) // 1)
    z = int((a * 10 ** (n - 1 - dec) + 0.5) // 1) / 10 ** (n - 1 - dec)
    return str(z) if z % 1 else str(int(z))


def print_elapsed_time():  # prints the simulation's elapsed time to five significant figures
    print('Total time elapsed: {}s'.format(adj_sig_figs(time.time() - start_time, 5)))


# prompts for config input
if len(sys.argv) > 1:
    config_file_name = str(sys.argv[1])
    print('Config file: {}'.format(config_file_name))
else:
    config_file_name = input('Config file (default: default-config): ')
    if config_file_name == '':
        config_file_name = 'default-config'
config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config-files', '{}.cfg'.format(config_file_name))
if not os.path.exists(config_file_path):
    print('That config file does not exist.')
    exit()
cfg = configparser.RawConfigParser()
cfg.read(config_file_path)
globals_dict = dict(cfg.items('Settings'))
for var in globals_dict:
    globals()[var.upper()] = eval(globals_dict[var].split('#', 1)[0].strip())
num_days = int(input('Number of days to collect data from (example: 7): '))

# static global variables
start_time = time.time()  # current real-world time
now = datetime.now()  # current real-world timestamp
daily_simulation_time = SIMULATION_TICKS_PER_HOUR * 24  # number of simulation ticks that occur each day
numpy.random.seed(RANDOM_SEED)  # used for both numpy and scipy
random.seed(RANDOM_SEED)
mallet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mallet-2.0.8', 'bin', 'mallet')
os.environ['MALLET_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mallet-2.0.8')

# checks locality type, independent cities (e.g. Baltimore City) are not currently compatible
if STATE_ABBR == 'AK':
    LOCALITY_TYPE = 'borough'
elif STATE_ABBR == 'LA':
    LOCALITY_TYPE = 'parish'
else:
    LOCALITY_TYPE = 'county'
locality_name = (LOCALITY + ' ' + LOCALITY_TYPE).title()  # e.g. "Fairfax County"
area = locality_name.lower().replace(' ', '-') + '-' + STATE_ABBR.lower()  # e.g. "fairfax-county-va"
weekly_patterns_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parsed-weekly-patterns', '{}-{}.csv'.format(area, WEEK))
if not os.path.exists(weekly_patterns_path):
    print('The data for the specified week and/or specified locatity has not been parsed yet. Please run `county_parser.py` before continuing.')
    exit()

# read in FIPS code prefix data
prefix_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_fips_codes.csv'), error_bad_lines=False)
prefix_row = prefix_data.loc[(prefix_data['county'] == locality_name) & (prefix_data['state'] == STATE_ABBR.upper())]
locality_prefix = str(prefix_row['state_fips'].item()).zfill(2) + str(prefix_row['county_fips'].item()).zfill(3)

print('Reading POI data...')

# read in data from weekly movement pattern files, filtering the data that corresponds to the area of interest
data = pd.read_csv(weekly_patterns_path, error_bad_lines=False)
usable_data = data[(data.visitor_home_cbgs != '{}')]

# from usable POIs, load cbgs, CBGs are documents, POIs are words
cbgs_to_pois = {}
dwell_distributions = {}
poi_set = set()
poi_type = {}
poi_hour_counts = {}  # poi id: length 24 list of poi visits by hour
coords = {}
for row in usable_data.itertuples():
    place_id = str(row.safegraph_place_id)
    poi_set.add(place_id)
    dwell_distributions[place_id] = eval(row.dwell_distribution)
    lati, longi = float(row.latitude), float(row.longitude)
    if not math.isnan(lati) and not math.isnan(longi):
        coords[place_id] = [lati, longi]
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
    cbg_frequencies = {}
    total_cbg_frequency = 0
    for cbg in cbgs:
        if not cbg.startswith(locality_prefix):  # excludes all CBGs outside of the locality of focus
            continue
        if cbg not in cbgs_to_pois:
            cbgs_to_pois[cbg] = []
        cbg_frequency = numpy.random.randint(2, 5) if cbgs[cbg] == 4 else cbgs[cbg]  # SafeGraph reports POI CBG frequency as 4 if the visit count is between 2-4
        total_cbg_frequency += cbg_frequency
        cbg_frequencies[cbg] = cbg_frequency
    raw_visit_count = float(row.raw_visit_counts)
    for cbg in cbg_frequencies:
        cbgs_to_pois[cbg].extend([place_id] * int(round(cbg_frequencies[cbg] / total_cbg_frequency * raw_visit_count)))  # generate POIs as "words" and multiply by the amount that CBG visited    

print_elapsed_time()
print('Running LDA and creating initial distributions...')

cbg_ids = list(cbgs_to_pois.keys())
cbg_id_set = set(cbg_ids)
lda_documents = list(cbgs_to_pois.values())
poi_set = {poi for poi_list in lda_documents for poi in poi_list}
poi_count = len(poi_set)

lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents)  # generate "documents" for gensim
lda_corpus = [lda_dictionary.doc2bow(cbg) for cbg in lda_documents]  # generate "words" for gensim
lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=lda_corpus, num_topics=NUM_TOPICS, id2word=lda_dictionary, random_seed=RANDOM_SEED, optimize_interval=1, alpha=50/NUM_TOPICS)

cbgs_to_topics = dict(zip(cbg_ids, list(lda_model.load_document_topics())))  # {'510594301011': [(0, 5.5570694e-05), (1, 5.5570694e-05), (2, 5.5570694e-05), (3, 5.5570694e-05), ...], ...}
lda_output = lda_model.show_topics(formatted=False, num_topics=NUM_TOPICS, num_words=poi_count)
topics_to_pois = [[[tup[0] for tup in givens[1]], [tup[1] for tup in givens[1]]] for givens in lda_output]  # [[poi id list, probability dist], [poi id list, probability dist], ...]

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
    cbgs_leaving_probs[cbg] = (1 - (cbg_device_counts[cbg][0] / cbg_device_counts[cbg][1])) * PROPENSITY_TO_LEAVE
print('Average percentage of population tracked per day: {}%'.format(adj_sig_figs(100 * total_devices_tracked / (total_population * 7))))

topic_numbers = [i for i in range(NUM_TOPICS)]

print('Preparing simulation... 0%')

# SETUP SIMULATION
# Generate agents for CBGs, create dictionary of active agents for each hour
# Create POI dictionary to store agents
# Randomly select a list of CBGs of the population size based on their respective probability to the population size
# Within that list of CBGs, iterate for each CBG
# Generate a randomly chosen topic based on the CBG percentage in each topic
# Once a topic and cbg are chosen, use a 0.1% probability to decide whether or not that agent is infected

agents = {}  # agent_id: [topic, infected_status, (current poi, expiration time) or None, home_cbg_code, other_household_member_set]
households = {}  # household_id: {agent_id, agent_id, ...}
cbgs_to_agents = {cbg: set() for cbg in cbg_ids}  # cbg: {agent_id, agent_id, ...}
inactive_agent_ids = set()  # {agent_id, agent_id, ...}
active_agent_ids = {}  # expiration_time: {(agent_id, poi_id), (agent_id, poi_id), ...}
poi_current_visitors = {poi: set() for poi_list in lda_documents for poi in poi_list}  # poi id: {agent_id, agent_id, ...}

agent_count = 0
household_count = 0


def add_agent(current_cbg, household_id, probs):
    global agent_count
    agent_id = 'agent_{}'.format(agent_count)
    agent_count += 1
    agent_topic = numpy.random.choice(topic_numbers, 1, p=probs)[0]
    agent_status = 'S'
    inactive_agent_ids.add(agent_id)
    agents[agent_id] = [agent_topic, agent_status, None, current_cbg, set()]
    cbgs_to_agents[current_cbg].add(agent_id)
    households[household_id].add(agent_id)


benchmark = 5
for i, current_cbg in enumerate(cbg_ids):
    probs = numpy.array(cbg_topic_probabilities[current_cbg])
    probs /= probs.sum()
    for idx, cbg_household_count in enumerate(cbgs_to_households[current_cbg]):
        current_household_size = idx % 7 + idx // 7 + 1  # nonfamily households, then family households
        for _ in range(cbg_household_count):
            household_id = 'household_{}'.format(household_count)
            household_count += 1
            households[household_id] = set()
            for _ in range(current_household_size):
                add_agent(current_cbg, household_id, probs)
            for agent_id in households[household_id]:
                agents[agent_id][4] = households[household_id] - {agent_id}
    completion = i / len(cbg_ids) * 100
    if completion >= benchmark:
        print('Preparing simulation... {}%'.format(adj_sig_figs(benchmark)))
        benchmark += 5

print_elapsed_time()
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
outfile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mobility-counts', config_file_name + '-' + '{}days'.format(num_days) + '.txt')
print('Running simulation... (check {} for the output)'.format(outfile_path))
outfile = open(outfile_path, 'a+')
sys.stdout = outfile
print('Number of agents: {}'.format(len(agents)))


def remove_expired_agents(current_time):  # sends agents home from POIs whose visits end at current_time
    global inactive_agent_ids, poi_current_visitors
    if current_time in active_agent_ids:  # checks if any agents are active
        for tup in active_agent_ids[current_time]:  # iterates through all active agents whose visits end at current_time
            poi_current_visitors[tup[1]].remove(tup[0])  # removes the agent from the POI
            inactive_agent_ids.add(tup[0])  # sends the agent home, allows the agent to visit more POIs
            agents[tup[0]][2] = None  # removes the POI the agent was at from the agent's data
        del active_agent_ids[current_time]  # deletes the now-useless set to conserve memory


def get_dwell_time(dwell_tuple):  # given a cached tuple from the dwell_distributions dictionary for a specific POI, return a dwell time in ticks
    dwell_time_minutes = 0  # represents dwell time in minutes (not ticks)
    if len(dwell_tuple) == 3:
        dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(loc=dwell_tuple[1], scale=dwell_tuple[2], size=NUMBER_OF_DWELL_SAMPLES))
    elif len(dwell_tuple) == 4:
        dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(dwell_tuple[1], loc=dwell_tuple[2], scale=dwell_tuple[3], size=NUMBER_OF_DWELL_SAMPLES))
    else:
        dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(dwell_tuple[1], dwell_tuple[2], loc=dwell_tuple[3], scale=dwell_tuple[4], size=NUMBER_OF_DWELL_SAMPLES))
    dwell_time_ticks = int(round(dwell_time_minutes * SIMULATION_TICKS_PER_HOUR / 60))  # represents dwell time in ticks
    if dwell_time_ticks < 1:  # minimum visit duration is one tick
        dwell_time_ticks = 1
    elif dwell_time_ticks > MAX_DWELL_TIME * SIMULATION_TICKS_PER_HOUR:  # maximum visit duration is MAX_DWELL_TIME * SIMULATION_TICKS_PER_HOUR ticks
        dwell_time_ticks = MAX_DWELL_TIME * SIMULATION_TICKS_PER_HOUR
    return dwell_time_ticks


def add_to_active_agent_ids(agent_id, destination, destination_end_time):  # adds agent_id and destination pair to active_agent_ids
    global active_agent_ids
    if not destination_end_time in active_agent_ids:
        active_agent_ids[destination_end_time] = {(agent_id, destination)}
    else:
        active_agent_ids[destination_end_time].add((agent_id, destination))


def select_active_agents(current_time):  # sends agents to POIs whose visits start at current_time
    global inactive_agent_ids, active_agent_ids, poi_current_visitors, total_poi_visit_counts
    hour = current_time // SIMULATION_TICKS_PER_HOUR % 24
    to_remove = set()  # agent ids to remove from inactive_agent_ids once iteration completes
    for agent_id in inactive_agent_ids:  # iterates through each agent at home that has the potential to leave the house
        topic = agents[agent_id][0]  # stores the agent's topic
        # cbgs_leaving_probs[agents[agent_id][3]] => the chance that a given agent will leave their house in the span of a day, determined by social distancing data from their CBG (typically ~75% in normal times)
        # topic_hour_distributions[topic][hour] => given that an agent will visit a CBG today, represents the chance that the visit will take place during this hour (~6% for typical daylight hours in normal times), determined by the hour of the day and the agent's topic
        # topic_hour_distributions[topic][hour] / SIMULATION_TICKS_PER_HOUR => given that an agent will visit a CBG today, represents the chance that the visit will take place during this tick, determined by the hour of the day and the agent's topic
        if numpy.random.rand() < cbgs_leaving_probs[agents[agent_id][3]] * topic_hour_distributions[topic][hour] / SIMULATION_TICKS_PER_HOUR:  # the agent has a (cbgs_leaving_probs[agents[agent_id][3]] * topic_hour_distributions[topic][hour] / SIMULATION_TICKS_PER_HOUR)% of leaving their house to visit a POI
            destination = numpy.random.choice(topics_to_pois_by_hour[topic][hour][0], 1, p=topics_to_pois_by_hour[topic][hour][1])[0]  # chooses a POI for the agent to visit based on the topic to POI distribution of the agent's topic for the specified hour
            if destination == 'aux':  # means the POI has a likelihood of being selected in the bottom 20%, simply used to speed up code
                destination = numpy.random.choice(aux_dict[topic][hour][0], 1, p=aux_dict[topic][hour][1])[0]  # chooses a POI from the auxiliary (bottom 20%) set for the agent to visit based on the topic to POI distribution of the agent's topic for the specified hour
            total_poi_visit_counts[destination] += 1
            destination_end_time = current_time + get_dwell_time(dwell_distributions[destination])  # time when the agent returns home
            add_to_active_agent_ids(agent_id, destination, destination_end_time)  # adds agent_id and destination pair to active_agent_ids
            poi_current_visitors[destination].add(agent_id)  # adds agent to POI
            agents[agent_id][2] = (destination, destination_end_time)  # agent's information now contains their current POI
            to_remove.add(agent_id)  # marks the agent_id for removal from inactive_agent_ids
    inactive_agent_ids -= to_remove  # all the agents who just traveled to POIs are no longer available for visits


def check_end_of_day(current_time):  # checks if current_time marks the end of a day, running special code if so
    if not (current_time + 1) % daily_simulation_time:  # if the next tick marks midnight
        day = current_time // daily_simulation_time
        print('Day {} complete.'.format(day))
        print('Day {}:'.format(day + 1))


def check_end_of_hour(current_time):  # checks if current_time marks the end of an hour, running special code if so
    if not (current_time + 1) % SIMULATION_TICKS_PER_HOUR:  # ensures the next tick marks the start of an hour
        print('Hour {} ({}) complete.'.format(int(current_time // SIMULATION_TICKS_PER_HOUR % 24), int(current_time / SIMULATION_TICKS_PER_HOUR)))
        print_elapsed_time()
        check_end_of_day(current_time)  # checks if current_time marks the end of a day, running special code if so
        update_outfile()


print('Day 0:')
update_outfile()
total_poi_visit_counts = {poi: 0 for poi in poi_set}
for current_time in range(num_days * daily_simulation_time):  # each iteration represents one simulation tick, terminates when nobody has an active infection
    remove_expired_agents(current_time)  # sends agents home from POIs whose visits end at current_time
    select_active_agents(current_time)  # sends agents to POIs whose visits start at current_time
    check_end_of_hour(current_time)  # checks if current_time marks the end of an hour, running special code if so

print()
print('Data collection complete!')
print('Final statistics:')
print(total_poi_visit_counts)
print_elapsed_time()
print()
outfile.close()

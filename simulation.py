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
import pickle
import statistics
import gzip
import queue
import scipy.stats
import configparser
import folium
from folium.plugins import HeatMap

# BEGIN DEFAULT CONFIGURATION FOR LINTER, EDITING THIS DOES NOTHING!!! USE .cfg FILES INSTEAD!!!

# preliminary parameters
LOCALITY = 'Fairfax'  # can be a county, borough, or parish; independent cities (e.g. Baltimore City) are not currently compatible
STATE_ABBR = 'VA'  # state abbreviation of where the locality is located
WEEK = '2019-10-28'  # start date of the week to base data off of, must be a Monday
NUM_TOPICS = 50  # number of Latent Dirichlet Allocation (LDA) topics
MAXIMUM_NUMBER_OF_MAPPED_POIS = 100  # the top MAXIMUM_NUMBER_OF_MAPPED_POIS for each topic will appear on the topic to poi map
RANDOM_SEED = 1  # SETTING THIS TO 0 USES THE SYSTEM CLOCK!! random seed for all random events in this script
PROPORTION_OF_POPULATION = 1  # 0.2 => 20% of the actual population is simulated, a number lower than 1 will cause the curve to "flatten" because all POIs are still simulated
PROPENSITY_TO_LEAVE = 1  # 0.2 => people are only 20% as likely to leave the house as compared to normal

# agent generation parameters
SIMULATION_TICKS_PER_HOUR = 4  # integer, number of ticks per simulation "hour"
ORIGIN_CBG = '510594804023'  # only agents in this CBG will begin with the virus, to include all CBGs use 'random'
PROPORTION_INITIALLY_INFECTED = 0.25  # 0.05 => 5% of the origin CBG (or the entire population if ORIGIN_CBG == 'random') is initially infected or exposed
ALPHA = 0  # 0.4 => 40% of the population is quarantined in their house for the duration of the simulation

# runtime parameters
MAX_DWELL_TIME = 16  # maximum dwell time at any POI (hours)
NUMBER_OF_DWELL_SAMPLES = 5  # a higher number decreases POI dwell time variation and allows less outliers
MAXIMUM_INTERACTIONS_PER_TICK = 5  # integer, maximum number of interactions an infected person can have with others per tick
MINIMUM_INTERVENTION_PROPORTION = 0  # 0.1 => all below interventions begin when 10% of the simulated population is infected
SYMPTOMATIC_QUARANTINES = False  # if True, quarantines all newly-infected agents upon showing symptoms after the MINIMUM_INTERVENTION_PROPORTION is reached
HOUSEHOLD_QUARANTINES = False  # if True, all household members will quarantine if an agent in their household also quarantines due to symptoms
QUARANTINE_DURATION = 10  # number of days a symptomatic-induced quarantine lasts
CLOSED_POI_TYPES = {}  # closes the following POI types (from SafeGraph Core Places "sub_category") after the MINIMUM_INTERVENTION_PROPORTION is reached

# virus parameters
# For COVID-19, a close contact is defined as ay individual who was within 6 feet of an infected person for at least 15 minutes starting from 2 days before illness onset (or, for asymptomatic patients, 2 days prior to positive specimen collection) until the time the patient is isolated. (https://www.cdc.gov/coronavirus/2019-ncov/php/contact-tracing/contact-tracing-plan/contact-tracing.html)
percent_asymptomatic = 0.4  # (recommended: 0.4) https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
secondary_attack_rate = 0.05  # (recommended: 0.05) chance of contracting the virus on close contact with someone, DO NOT DIVIDE BY SIMULATION_TICKS_PER_HOUR https://jamanetwork.com/journals/jama/fullarticle/2768396
asymptomatic_relative_infectiousness = 0.75  # (recommended: 0.75) https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
distribution_of_exposure = scipy.stats.gamma(4, 0, 0.75)  # (recommended: 4, 0, 0.75) gamma distribution of the duration (days) between exposure and infectiousness, k=4 μ=3 => midpoint is 2.754 days https://www.nature.com/articles/s41591-020-0962-9
distribution_of_preclinical = scipy.stats.gamma(4, 0, 0.525)  # (recommended: 4, 0, 0.525) gamma distribution of the duration (days) between infectiousness and symptoms for symptomatic cases, k=4 μ=2.1 => midpoint is 1.928 days https://www.nature.com/articles/s41591-020-0962-9
distribution_of_clinical = scipy.stats.gamma(4, 0, 0.725)  # (recommended: 4, 0, 0.725) gamma distribution of the duration (days) between symptoms and non-infectiousness (recovery) for symptomatic cases, k=4 μ=2.9 => midpoint is 2.662 days https://www.nature.com/articles/s41591-020-0962-9
distribution_of_subclinical = scipy.stats.gamma(4, 0, 1.25)  # (recommended: 4, 0, 1.25) gamma distribution of the duration (days) between infectiousness and non-infectiousness (recovery) for asymptomatic cases, k=4 μ=5 => midpoint is 4.590 days https://www.nature.com/articles/s41591-020-0962-9
total_chance_of_small_household_transmission = 0.204  # (recommended: 0.091) chance of an infected agent spreading the virus to a given household member over the agent's entire period of infection when the household size is six or less https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30471-0/fulltext
total_chance_of_large_household_transmission = 0.091  # (recommended: 0.204) chance of an infected agent spreading the virus to a given household member over the agent's entire period of infection when the household size is more than six https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30471-0/fulltext

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


def create_map(data, map_path, map_lat=38.8300, map_long=-77.2800, zoom_level=11):  # create heatmap [[lat, long, weight], [lat, long, weight], ... [lat, long, weight]]
        folium_map = folium.Map([map_lat, map_long], tiles='stamentoner', zoom_start=11)
        for idx, topic in enumerate(data):
            h = HeatMap(topic, show=False)
            h.layer_name = 'topic_{}'.format(idx)
            folium_map.add_child(h)
        folium.LayerControl(collapsed=False).add_to(folium_map)
        folium_map.save(map_path)


# prompt user whether or not to used cached data in order to reduce initial simulation load time
raw_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_cache.p')  # cache path of the SafeGraph and LDA data
agents_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents_cache.p')  # cache path of the agent data
agents_loaded = False
if os.path.exists(raw_cache_path) and not len(sys.argv) > 1:
    use_raw_cache = input('Use file data cache (y/[n])? ')
else:
    use_raw_cache = 'n'
if use_raw_cache.lower() == 'y':  # obtains cached variables from the file data cache
    raw_cache_file = open(raw_cache_path, 'rb')
    (config_file_name, cbg_ids, lda_documents, cbgs_to_households, cbg_topic_probabilities, topics_to_pois, cbgs_leaving_probs, dwell_distributions, poi_type, topic_hour_distributions, topics_to_pois_by_hour) = pickle.load(raw_cache_file)
    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config-files', '{}.cfg'.format(config_file_name))
    cfg = configparser.RawConfigParser()
    cfg.read(config_file_path)
    globals_dict = dict(cfg.items('Settings'))
    for var in globals_dict:
        globals()[var.upper()] = eval(globals_dict[var].split('#', 1)[0].strip())
    # static global variables
    start_time = time.time()  # current real-world time
    now = datetime.now()  # current real-world timestamp
    daily_simulation_time = SIMULATION_TICKS_PER_HOUR * 24  # number of simulation ticks that occur each day
    numpy.random.seed(RANDOM_SEED)  # used for both numpy and scipy
    random.seed(RANDOM_SEED)
    exposure_median = distribution_of_exposure.median()
    preclinical_median = distribution_of_preclinical.median()
    clinical_median = distribution_of_clinical.median()
    subclinical_median = distribution_of_subclinical.median()
    median_infectious_duration = preclinical_median + clinical_median
    daily_chance_of_small_household_transmission = total_chance_of_small_household_transmission / median_infectious_duration
    daily_chance_of_large_household_transmission = total_chance_of_large_household_transmission / median_infectious_duration
    mallet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mallet-2.0.8', 'bin', 'mallet')
    os.environ['MALLET_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mallet-2.0.8')
    if os.path.exists(agents_cache_path):
        use_agents_cache = input('Use agents cache (y/[n])? ')  # obtains cached variables from agent file data cache
    else:
        use_agents_cache = 'n'
    if use_agents_cache.lower() == 'y':
        agents_cache_file = open(agents_cache_path, 'rb')
        (agents, households, cbgs_to_agents, inactive_agent_ids, active_agent_ids, poi_current_visitors, Ipa_queue_tups, Ic_queue_tups, R_queue_tups, quarantine_queue_tups, total_ever_infected) = pickle.load(agents_cache_file)
        agents_loaded = True
else:  # loads and caches data from files depending on user input
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
    
    # static global variables
    start_time = time.time()  # current real-world time
    now = datetime.now()  # current real-world timestamp
    daily_simulation_time = SIMULATION_TICKS_PER_HOUR * 24  # number of simulation ticks that occur each day
    numpy.random.seed(RANDOM_SEED)  # used for both numpy and scipy
    random.seed(RANDOM_SEED)
    exposure_median = distribution_of_exposure.median()
    preclinical_median = distribution_of_preclinical.median()
    clinical_median = distribution_of_clinical.median()
    subclinical_median = distribution_of_subclinical.median()
    median_infectious_duration = preclinical_median + clinical_median
    daily_chance_of_small_household_transmission = total_chance_of_small_household_transmission / median_infectious_duration
    daily_chance_of_large_household_transmission = total_chance_of_large_household_transmission / median_infectious_duration
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
    poi_types = set()
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
        poi_types.add(place_type)
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
    
    print(poi_types)
    exit()
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
    print('Creating topic to POI map...')

    topic_to_poi_map_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'topic-to-poi-maps', config_file_name + '.html')
    map_data = [[coords[tup[0]] + [float(tup[1])] for tup in givens[1] if tup[0] in coords][:min(len(givens[1]), MAXIMUM_NUMBER_OF_MAPPED_POIS)] for givens in lda_output]
    average_lat = sum([sum([tup[0] for tup in topic_info]) for topic_info in map_data]) / sum([len(topic_info) for topic_info in map_data])
    average_long = sum([sum([tup[1] for tup in topic_info]) for topic_info in map_data]) / sum([len(topic_info) for topic_info in map_data])
    create_map(map_data, topic_to_poi_map_path, average_lat, average_long)

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

    print_elapsed_time()
    print('Caching raw data...')

    raw_cache_data = (config_file_name, cbg_ids, lda_documents, cbgs_to_households, cbg_topic_probabilities, topics_to_pois, cbgs_leaving_probs, dwell_distributions, poi_type, topic_hour_distributions, topics_to_pois_by_hour)
    raw_cache_file = open(raw_cache_path, 'wb')
    pickle.dump(raw_cache_data, raw_cache_file)

    print_elapsed_time()

raw_cache_file.close()

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

    agents = {}  # agent_id: [topic, infected_status, (current poi, expiration time) or None, home_cbg_code, other_household_member_set]
    households = {}  # household_id: {agent_id, agent_id, ...}
    cbgs_to_agents = {cbg: set() for cbg in cbg_ids}  # cbg: {agent_id, agent_id, ...}
    inactive_agent_ids = set()  # {agent_id, agent_id, ...}
    active_agent_ids = {}  # expiration_time: {(agent_id, poi_id), (agent_id, poi_id), ...}
    poi_current_visitors = {poi: set() for poi_list in lda_documents for poi in poi_list}  # poi id: {agent_id, agent_id, ...}
    total_ever_infected = 0
    Ipa_queue_tups = []
    Ic_queue_tups = []
    R_queue_tups = []
    quarantine_queue_tups = []

    agent_count = 0
    household_count = 0


    def add_agent(current_cbg, household_id, probs):
        global agent_count, total_ever_infected
        agent_id = 'agent_{}'.format(agent_count)
        agent_count += 1
        agent_topic = numpy.random.choice(topic_numbers, 1, p=probs)[0]
        agent_status = 'S'
        permanently_quarantined = numpy.random.rand() < ALPHA  # prohibits agent from ever leaving their house and ensures they are not initially infected if True
        parameter_2 = None
        if ORIGIN_CBG.lower() in {current_cbg, 'random'} and not permanently_quarantined:
            rand = numpy.random.rand()
            if rand < PROPORTION_INITIALLY_INFECTED:
                rand /= PROPORTION_INITIALLY_INFECTED
                total_ever_infected += 1
                E_bound = exposure_median / (exposure_median + preclinical_median + clinical_median + subclinical_median)  # represents chance an infected agent is exposed and not contagious or symptomatic yet
                Ia_bound = E_bound + percent_asymptomatic * (1 - E_bound)  # percent_asymptomatic% of the remaining value, represents asymptomatic (subclinical) cases
                Ip_bound = Ia_bound + preclinical_median / median_infectious_duration * (1 - Ia_bound)  # represents preclinical cases
                if rand < E_bound:
                    agent_status = 'E'
                    Ipa_queue_tups.append((round(distribution_of_exposure.rvs(1)[0] * daily_simulation_time), agent_id))
                    inactive_agent_ids.add(agent_id)
                elif rand < Ia_bound:
                    agent_status = 'Ia'
                    R_queue_tups.append((round(distribution_of_subclinical.rvs(1)[0] * daily_simulation_time), agent_id))
                    inactive_agent_ids.add(agent_id)
                elif rand < Ip_bound:
                    agent_status = 'Ip'
                    Ic_queue_tups.append((round(distribution_of_preclinical.rvs(1)[0] * daily_simulation_time), agent_id))
                    inactive_agent_ids.add(agent_id)
                else:  # represents symptomatic (clinical) cases
                    agent_status = 'Ic'
                    R_queue_tups.append((round(distribution_of_clinical.rvs(1)[0] * daily_simulation_time), agent_id))
                    if SYMPTOMATIC_QUARANTINES and PROPORTION_INITIALLY_INFECTED >= MINIMUM_INTERVENTION_PROPORTION:
                        quarantine_queue_tups.append((QUARANTINE_DURATION * 24 * SIMULATION_TICKS_PER_HOUR, agent_id))
                        parameter_2 = 'quarantined'
                    else:
                        inactive_agent_ids.add(agent_id)
            else:
                inactive_agent_ids.add(agent_id)
        else:
            inactive_agent_ids.add(agent_id)
        agents[agent_id] = [agent_topic, agent_status, parameter_2, current_cbg, set()]
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
                    if HOUSEHOLD_QUARANTINES and SYMPTOMATIC_QUARANTINES and PROPORTION_INITIALLY_INFECTED >= MINIMUM_INTERVENTION_PROPORTION and agents[agent_id][2] == 'quarantined':
                        for other_agent_id in agents[agent_id][4]:
                            if agents[other_agent_id][2] != 'quarantined':
                                quarantine_queue_tups.append((QUARANTINE_DURATION * 24 * SIMULATION_TICKS_PER_HOUR, other_agent_id))
                                agents[other_agent_id][2] = 'quarantined'
        completion = i / len(cbg_ids) * 100
        if completion >= benchmark:
            print('Preparing simulation... {}%'.format(adj_sig_figs(benchmark)))
            benchmark += 5
    
    print_elapsed_time()
    print('Creating CBG to topic map...')
    
    cbg_to_topic_map_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'cbg-to-topic-maps', config_file_name + '.html')
    map_cbg_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_geographic_data.csv'))
    topic_cbg_counts = {i: {} for i in range(NUM_TOPICS)}
    for agent in agents:
        if agents[agent][3] not in topic_cbg_counts[agents[agent][0]]:
            topic_cbg_counts[agents[agent][0]][agents[agent][3]] = 1
        else:
            topic_cbg_counts[agents[agent][0]][agents[agent][3]] += 1
    map_data = []
    for topic in [sorted([(topic_cbg_counts[key][cbg], cbg) for cbg in topic_cbg_counts[key]]) for key in topic_cbg_counts]:
        topic_data = []
        topic_population = sum([tup[0] for tup in topic])
        for iu, tup in enumerate(topic):
            row = map_cbg_data.loc[map_cbg_data.census_block_group == int(tup[1])]
            topic_data.append([float(row.latitude), float(row.longitude), float(tup[0])])
        map_data.append(topic_data)
    average_lat = sum([sum([tup[0] for tup in topic_info]) for topic_info in map_data]) / sum([len(topic_info) for topic_info in map_data])
    average_long = sum([sum([tup[1] for tup in topic_info]) for topic_info in map_data]) / sum([len(topic_info) for topic_info in map_data])
    create_map(map_data, cbg_to_topic_map_path, average_lat, average_long)

    print_elapsed_time()
    print('Caching agents data...')

    agents_cache_data = (agents, households, cbgs_to_agents, inactive_agent_ids, active_agent_ids, poi_current_visitors, Ipa_queue_tups, Ic_queue_tups, R_queue_tups, quarantine_queue_tups, total_ever_infected)
    agents_cache_file = open(agents_cache_path, 'wb')
    pickle.dump(agents_cache_data, agents_cache_file)

    print_elapsed_time()

agents_cache_file.close()
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
outfile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'console-outputs', config_file_name + '.txt')
print('Running simulation... (check {} for the output)'.format(outfile_path))
outfile = open(outfile_path, 'a+')
sys.stdout = outfile
print('Number of agents: {}'.format(len(agents)))
print('Total ever infected: {} ({}%)'.format(total_ever_infected, adj_sig_figs(100 * total_ever_infected / len(agents))))


def check_interventions(current_time):  # checks if the intervention threshold has been met, deploys interventions if so
    global interventions_deployed, closed_pois, cbgs_leaving_probs, do_quarantines
    if not interventions_deployed and total_ever_infected / len(agents) >= MINIMUM_INTERVENTION_PROPORTION:
        print('THE POPULATION THRESHOLD FOR INTERVENTIONS HAS BEEN MET! DEPLOYING INTERVENTIONS...')
        interventions_deployed = True
        for current_poi_type in CLOSED_POI_TYPES:  # e.g. "Restaurants and Other Eating Places"
            closed_pois |= poi_type[current_poi_type]
        if SYMPTOMATIC_QUARANTINES:  # symptomatic quarantines
            do_quarantines = True


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
    global inactive_agent_ids, active_agent_ids, poi_current_visitors
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
            if destination in closed_pois:  # desired destination is closed, agent forgoes the visit and stays home
                continue
            destination_end_time = current_time + get_dwell_time(dwell_distributions[destination])  # time when the agent returns home
            add_to_active_agent_ids(agent_id, destination, destination_end_time)  # adds agent_id and destination pair to active_agent_ids
            poi_current_visitors[destination].add(agent_id)  # adds agent to POI
            agents[agent_id][2] = (destination, destination_end_time)  # agent's information now contains their current POI
            to_remove.add(agent_id)  # marks the agent_id for removal from inactive_agent_ids
    inactive_agent_ids -= to_remove  # all the agents who just traveled to POIs are no longer available for visits


def infect(agent_id, current_time):  # infects an agent with the virus if they are susceptible
    global infection_counts_by_day, total_ever_infected, Ipa_queue
    if agents[agent_id][1] == 'S':  # ensures an agent is susceptible
        agents[agent_id][1] = 'E'  # marks the agent as exposed
        Ipa_queue.put((current_time + round(distribution_of_exposure.rvs(1)[0] * daily_simulation_time), agent_id))  # puts the agent in a queue to become either asymptomatic or presymptomatic in a few days
        infection_counts_by_day[current_time // daily_simulation_time] += 1  # increments the daily infection count
        total_ever_infected += 1  # increments the total infection count


def possibly_infect(contacted_list, current_time, relative_infectiousness):  # possibly infects a list of agents an infected agent had contact with
    for contacted_agent_id in contacted_list:  # iterates through each agent an infected agent had direct contact with
        if numpy.random.rand() < secondary_attack_rate * relative_infectiousness:  # the contacted agent has a (secondary_attack_rate * relative_infectiousness)% chance of becoming infected if they are susceptible
            infect(contacted_agent_id, current_time)  # infects the contacted agent with the virus if they are susceptible


def infect_active_agents(current_time):  # iterates through each POI, allowing any visiting agents to spread the virus to others
    for poi in poi_current_visitors:  # iterates through each POI
        if poi not in closed_pois and len(poi_current_visitors[poi]) > 1:  # ensures POI is not closed and that there is more than one current visitor
            poi_agents = list(poi_current_visitors[poi])  # creates a list of agent ids representing all of the POI's current visitors
            max_interactions = min(len(poi_agents) - 1, MAXIMUM_INTERACTIONS_PER_TICK)  # the maximum number of interactions any agent can have this tick in this POI
            for agent_idx, agent_id in enumerate(poi_agents):  # iterates through each visiting agent in the POI
                if agents[agent_id][1] == 'Ia' or agents[agent_id][1] == 'Ip':  # if an agent is presymptomatic or asymptomatic, they may spread the virus to others with a reduced chance of infection
                    possible_agents = poi_agents[:agent_idx] + poi_agents[agent_idx+1:]  # prohibits the infected agent from having contact with themself
                    possibly_infect(numpy.random.choice(possible_agents, max_interactions), current_time, asymptomatic_relative_infectiousness)  # the infected agent has contact with max_interactions other agents in the POI, possibly spreading the virus to them
                elif agents[agent_id][1] == 'Ic':  # if an agent is symptomatic, they may spread the virus to others with the full chance of infection
                    possible_agents = poi_agents[:agent_idx] + poi_agents[agent_idx+1:]  # prohibits the infected agent from having contact with themself
                    possibly_infect(numpy.random.choice(possible_agents, max_interactions), current_time, 1)  # the infected agent has contact with max_interactions other agents in the POI, possibly spreading the virus to them


def quarantine(agent, days_to_be_quarantined, current_time):  # keeps an agent in their house for days_to_be_quarantined days from current_time, the agent can still infect others in their household if they are infected
    global quarantine_queue, inactive_agent_ids
    if agents[agent][2] == 'quarantined':  # ensures the agent is not already quarantined
        return
    quarantine_queue.put((current_time + days_to_be_quarantined * daily_simulation_time, agent))  # the quarantine expires at tick current_time + days_to_be_quarantined * daily_simulation_time
    if agents[agent][2] == None:  # means the agent is currently home and not under quarantine already
        inactive_agent_ids.remove(agent)  # prohibits the agent from visiting POIs
    else:  # means the agent is currently at a POI
        visit_info = agents[agent][2]  # contains information on where the agent currently is
        poi_current_visitors[visit_info[0]].remove(agent)  # removes the agent from their current POI
        active_agent_ids[visit_info[1]].remove((agent, visit_info[0]))  # removes the agent from the active_agent_ids dictionary
    agents[agent][2] = 'quarantined'  # marks the agent as quarantined


def update_agent_status(current_time):  # updates agent infection and quarantine status
    global agents, Ipa_queue, Ic_queue, R_queue, quarantine_queue
    while Ipa_queue.qsize() and Ipa_queue.queue[0][0] <= current_time + 1:  # updates agents who are currently exposed to the virus and noncontagious
        key = Ipa_queue.get()[1]  # removes an agent from the queue
        if numpy.random.rand() < percent_asymptomatic:  # asymptomatic (subclinical) cases, never show symptoms
            agents[key][1] = 'Ia'  # marks the agent as asymptomatic, is now contagious at a reduced amount
            R_queue.put((current_time + round(distribution_of_subclinical.rvs(1)[0] * daily_simulation_time), key))  # puts the agent in queue for recovery (end of being contagious)
        else:  # preclinical cases, will be symptomatic in the future
            agents[key][1] = 'Ip'  # marks the agent as presymptomatic, is now contagious at a reduced amount
            Ic_queue.put((current_time + round(distribution_of_preclinical.rvs(1)[0] * daily_simulation_time), key))  # puts the agent in queue for symptoms (fully contagiousness then)
    while Ic_queue.qsize() and Ic_queue.queue[0][0] <= current_time + 1:  # updates agents who are currently presymptomatic and not fully contagious
        key = Ic_queue.get()[1]  # removes an agent from the queue
        agents[key][1] = 'Ic'  # marks the agent as symptomatic, is now fully contagious
        if do_quarantines:  # checks if the quarantine intervention is enabled
            quarantine(key, QUARANTINE_DURATION, current_time)  # quarantine the agent in their house for QUARANTINE_DURATION days
            if HOUSEHOLD_QUARANTINES:  # checks if quarantines for household members are enabled
                for household_member_id in agents[key][2]:  # iterates through each household member excluding the key agent
                    quarantine(household_member_id, QUARANTINE_DURATION, current_time)  # quarantines each household member in the house for QUARANTINE_DURATION days
        R_queue.put((current_time + round(distribution_of_clinical.rvs(1)[0] * daily_simulation_time), key))  # puts the agent in queue for recovery (end of being contagious)
    while R_queue.qsize() and R_queue.queue[0][0] <= current_time + 1:  # updates agents who are either symptomatic or asymptomatic and due for recovery (end of being contagious)
        key = R_queue.get()[1]  # removes an agent from the queue
        agents[key][1] = 'R'  # marks the agent as recovered, the agent can no longer spread the virus or become infected
    if do_quarantines:  # checks if the quarantine intervention is enabled
        while quarantine_queue.qsize() and quarantine_queue.queue[0][0] <= current_time + 1:  # updates agents who are currently quarantined and due to come out
            key = quarantine_queue.get()[1]  # removes an agent from the queue
            inactive_agent_ids.add(key)  # allows the agent to visit POIs again
            agents[key][2] = None  # removes quarantine status from the agent


def household_transmission(current_time):  # simulates a day's worth of of household transmission in every household
    for household in households.values():  # iterates through each household
        base_transmission_probability = daily_chance_of_small_household_transmission if len(household) <= 6 else daily_chance_of_large_household_transmission  # the probability an agent will spread the virus to another agent in their household today
        for agent_id in household:  # iterates through each agent in the household
            if agents[agent_id][1] == 'Ia' or agents[agent_id][1] == 'Ip':  # if an agent is presymptomatic or asymptomatic, they may spread the virus to others with a reduced chance of infection
                for other_agent_id in household:  # iterates through each other household member
                    if numpy.random.rand() < base_transmission_probability * asymptomatic_relative_infectiousness:  # the other household member has a (base_transmission_probability * asymptomatic_relative_infectiousness)% chance of becoming infected if they are susceptible
                        infect(other_agent_id, current_time)
            elif agents[agent_id][1] == 'Ic':  # if an agent is symptomatic, they may spread the virus to others with the full chance of infection
                for other_agent_id in household:  # iterates through each other household member
                    if numpy.random.rand() < base_transmission_probability:
                        infect(other_agent_id, current_time)


def report_status():  # reports the aggregate infection status of all agents
    global infection_counts_by_day
    stat = {'S': 0, 'E': 0, 'Ia': 0, 'Ip': 0, 'Ic': 0, 'R': 0}
    for tup in agents.values():  # iterates through each agent, checking their infection status
        stat[tup[1]] += 1
    print('Susceptible: {}'.format(stat['S']))
    print('Exposed: {}'.format(stat['E']))
    print('Infected: {}'.format(stat['Ia'] + stat['Ip'] + stat['Ic']))
    print('Recovered: {}'.format(stat['R']))
    print('New cases per day: {}'.format(list(zip([i for i in range(len(infection_counts_by_day))], infection_counts_by_day))))
    print(r'Total ever infected: {} ({}% of the total population)'.format(total_ever_infected, adj_sig_figs(100 * total_ever_infected / len(agents))))
    print()


def check_end_of_day(current_time):  # checks if current_time marks the end of a day, running special code if so
    global infection_counts_by_day
    if not (current_time + 1) % daily_simulation_time:  # if the next tick marks midnight
        day = current_time // daily_simulation_time
        household_transmission(current_time)  # agents can transmit the virus to other members of their household at the end of the day
        print('Daily household transmission complete.')
        print_elapsed_time()
        print('Day {} complete. Infection status:'.format(day))
        report_status()  # reports the aggregate infection status of all agents
        print('Day {}:'.format(day + 1))
        infection_counts_by_day.append(0)


def check_end_of_hour(current_time):  # checks if current_time marks the end of an hour, running special code if so
    if not (current_time + 1) % SIMULATION_TICKS_PER_HOUR:  # ensures the next tick marks the start of an hour
        print('Hour {} ({}) complete.'.format(int(current_time // SIMULATION_TICKS_PER_HOUR % 24), int(current_time / SIMULATION_TICKS_PER_HOUR)))
        print_elapsed_time()
        check_end_of_day(current_time)  # checks if current_time marks the end of a day, running special code if so
        update_outfile()


print('Day 0:')
update_outfile()
infection_counts_by_day = [0]  # increments on the day of EXPOSURE to the virus, any real-world scenario will lag behind due to testing logistics
interventions_deployed = False  # if True, check_interventions will do nothing
do_quarantines = False  # if True, symptomatic quarantines are enabled
closed_pois = set()  # set of all closed POI ids
current_time = 0  # current simulation time (ticks)
while Ipa_queue.qsize() or Ic_queue.qsize() or R_queue.qsize():  # each iteration represents one simulation tick, terminates when nobody has an active infection
    check_interventions(current_time)  # checks if the intervention threshold has been met, deploys interventions if so
    remove_expired_agents(current_time)  # sends agents home from POIs whose visits end at current_time
    select_active_agents(current_time)  # sends agents to POIs whose visits start at current_time
    infect_active_agents(current_time)  # iterates through each POI, allowing any visiting agents to spread the virus to others
    update_agent_status(current_time)  # updates agent infection and quarantine status
    check_end_of_hour(current_time)  # checks if current_time marks the end of an hour, running special code if so
    current_time += 1  # increments the current simulation time (ticks)

print()
print('Nobody is exposed or infected, simulation complete!')
print('Final statistics:')
report_status()  # reports the aggregate infection status of all agents
print_elapsed_time()
print()
outfile.close()

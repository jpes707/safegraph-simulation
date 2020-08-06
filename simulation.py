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
from distfit import distfit
import scipy.stats

NUM_TOPICS = 50  # number of LDA topics
SIMULATION_DAYS = 365 * 5  # number of "days" the simulation runs for
SIMULATION_HOURS_PER_DAY = 24  # number of "hours" per "day" in which agents visit POIs
SIMULATION_TICKS_PER_HOUR = 2  # number of ticks per simulation "hour"
PROPORTION_OF_POPULATION = 1  # 0.2 => 20% of the actual population is simulated
PROPORTION_INITIALLY_INFECTED = 0.001  # 0.05 => 5% of the simulated population is initially infected or exposed
PROPENSITY_TO_LEAVE = 1  # 0.2 => people are only 20% as likely to leave the house as compared to normal
NUMBER_OF_DWELL_SAMPLES = 5  # a higher number decreases POI dwell time variation and allows less outliers
QUARANTINE_DURATION = 10  # number of days a quarantine lasts after an agent begins to show symptoms
MAXIMUM_INTERACTIONS_PER_TICK = 5  # maximum number of interactions an infected person can have with others per tick
ALPHA = 0  # 0.4 => 40% of the population is quarantined in their house for the duration of the simulation
# MAXIMUM_REROLLS = 3  # maximum number of times people try to go to an alternative place if a POI is closed
CLOSED_POI_TYPES = {  # closed POI types (from SafeGraph Core Places "sub_category")
    'Full-Service Restaurants',
    'Limited-Service Restaurants',
    'Department Stores',
    'Musical Instrument and Supplies Stores',
    'Elementary and Secondary Schools',
    'Colleges, Universities, and Professional Schools',
    'Tobacco Stores',
    'Department Stores',
    'Snack and Nonalcoholic Beverage Bars',
    'Jewelry Stores',
    'Religious Organizations',
    'Florists',
    'Paint and Wallpaper Stores',
    'Women\'s Clothing Stores',
    'Fitness and Recreational Sports Centers',
    'Family Clothing Stores',
    'Junior Colleges',
    'Hobby, Toy, and Game Stores',
    'Gift, Novelty, and Souvenir Stores',
    'Book Stores',
    'Sporting Goods Stores',
    'Cosmetics, Beauty Supplies, and Perfume Stores',
    'Art Dealers',
    'Bowling Centers',
    'Beauty Salons',
    'Libraries and Archives',
    'Sewing, Needlework, and Piece Goods Stores',
    'Wineries',
    'Video Tape and Disc Rental',
    'Museums',
    'Children\'s and Infants\' Clothing Stores',
    'Motion Picture Theaters (except Drive-Ins)',
    'Shoe Stores',
    'Golf Courses and Country Clubs',
    'Furniture Stores',
    'Breweries',
    'Investment Advice',
    'Historical Sites',
    'Floor Covering Stores',
    'Carpet and Upholstery Cleaning Services',
    'Drinking Places (Alcoholic Beverages)',
    'Exam Preparation and Tutoring',
    'Boat Dealers'
}

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


# prompt user whether or not to used cached data in order to reduce initial simulation load time
raw_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_cache.p')
agents_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents_cache.p')
agents_loaded = False
use_raw_cache = input('Use file data cache ([y]/n)? ')
if use_raw_cache == '' or use_raw_cache == 'y':  # obtains cached variables from the file data cache
    raw_cache_file = open(raw_cache_path, 'rb')
    (cbg_ids, poi_set, hourly_lda, cbgs_to_age_distribution, cbgs_to_households, cbgs_leaving_probs, dwell_distributions, poi_type) = pickle.load(raw_cache_file)
    use_agents_cache = input('Use agents cache ([y]/n)? ')  # obtains cached variables from agent file data cache
    if use_agents_cache == '' or use_agents_cache == 'y':
        agents_cache_file = open(agents_cache_path, 'rb')
        (agents, households, cbgs_to_agents, inactive_agent_ids, quarantined_agent_ids, active_agent_ids, poi_current_visitors) = pickle.load(agents_cache_file)
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
    cbgs_to_pois = {k: {} for k in range(168)}
    dwell_distributions = {}
    cbgs_to_places = {}
    poi_set = set()
    poi_type = {}
    for row in usable_data.itertuples():
        place_id = str(row.safegraph_place_id)
        poi_set.add(place_id)
        dwell_distributions[place_id] = eval(row.dwell_distribution)
        place_type = str(row.sub_category)
        if place_type not in poi_type:
            poi_type[place_type] = {place_id}
        else:
            poi_type[place_type].add(place_id)
        hourly_poi = json.loads(row.visits_by_each_hour)
        weekly_poi = int(row.raw_visit_counts)
        cbgs = json.loads(row.visitor_home_cbgs)
        lda_words = []
        for cbg in cbgs:
            if not cbg.startswith('51059'):  # excludes all POIs outside of Fairfax County, must be removed later!
                continue
            cbg_frequency = random.randint(2, 4) if cbgs[cbg] == 4 else cbgs[cbg]  # SafeGraph reports POI CBG frequency as 4 if the visit count is between 2-4
            for hour, freq in enumerate(hourly_poi):
                if cbg not in cbgs_to_pois[hour]:
                    cbgs_to_pois[hour][cbg] = []
                cbgs_to_pois[hour][cbg].extend([place_id] * cbg_frequency * round(10000 * freq/weekly_poi))
            # cbgs_to_pois[cbg].extend([place_id] * cbg_frequency)  # generate POIs as "words" and multiply by the amount that CBG visited
    
    print_elapsed_time()
    print('Running LDA...')
    cbg_ids = list(cbgs_to_pois[0].keys())
    cbg_id_set = set(cbg_ids)

    hourly_lda = {k: [] for k in range(168)}
    for hour in range(168): # length of a week, in hours
        lda_documents = list(cbgs_to_pois[hour].values())
        poi_set = {poi for poi_list in lda_documents for poi in poi_list}
        poi_count = len(poi_set)

        lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents)  # generate "documents" for gensim
        lda_corpus = [lda_dictionary.doc2bow(cbg) for cbg in lda_documents]  # generate "words" for gensim
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

        hourly_lda[hour] = [lda_documents, cbgs_to_topics, topics_to_pois, cbg_topic_probabilities]

    print_elapsed_time()
    print('Reading population data...')

    census_population_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b00.csv'), error_bad_lines=False)
    cbgs_to_populations = {}
    total_population = 0
    for idx, row in census_population_data.iterrows():
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
    print('Reading household data...')

    census_household_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b11.csv'), error_bad_lines=False)
    cbgs_to_households = {}  # {cbg: [# nonfamily households of size 1, # nonfamily households of size 2, ... # nonfamily households of size 7+, # family households of size 2, # family households of size 3, ... # family households of size 7+]}
    total_households = 0
    for idx, row in census_household_data.iterrows():
        check_str = str(int(row['census_block_group'])).zfill(12)
        if check_str in cbg_id_set:
            arr = []
            for ext in range(10, 17):  # appends nonfamily household counts
                arr.append(int(int(row['B11016e{}'.format(ext)]) * PROPORTION_OF_POPULATION))
            for ext in range(3, 9):  # appends family household counts
                arr.append(int(int(row['B11016e{}'.format(ext)]) * PROPORTION_OF_POPULATION))
            total_households += sum(arr)
            cbgs_to_households[check_str] = arr

    print_elapsed_time()
    print('Reading age data...')

    census_age_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b01.csv'), error_bad_lines=False)
    cbgs_to_age_distribution = {}  # {cbg: [[% 18-49, % 50-64, % 65+], [% <18, % 18-49, % 50-64, % 65+]], ...} (first array is distribution for nonfamily households because nonfamily households cannot include age <18, second array is distribution for family households)
    code_under_18 = ['B01001e3', 'B01001e4', 'B01001e5', 'B01001e6', 'B01001e27', 'B01001e28', 'B01001e29', 'B01001e30']
    code_18_to_49 = ['B01001e7', 'B01001e8', 'B01001e9', 'B01001e10', 'B01001e11', 'B01001e12', 'B01001e13', 'B01001e14', 'B01001e15', 'B01001e31', 'B01001e32', 'B01001e33', 'B01001e34', 'B01001e35', 'B01001e36', 'B01001e37', 'B01001e38', 'B01001e39']
    code_50_to_64 = ['B01001e16', 'B01001e17', 'B01001e18', 'B01001e19', 'B01001e40', 'B01001e41', 'B01001e42', 'B01001e43']
    code_65_plus = ['B01001e20', 'B01001e21', 'B01001e22', 'B01001e23', 'B01001e24', 'B01001e25', 'B01001e44', 'B01001e45', 'B01001e46', 'B01001e47', 'B01001e48', 'B01001e49']
    for idx, row in census_age_data.iterrows():
        check_str = str(int(row['census_block_group'])).zfill(12)
        if check_str in cbg_id_set:
            pop_under_18 = sum([int(row[a]) for a in code_under_18])
            pop_18_49 = sum([int(row[a]) for a in code_18_to_49])
            pop_50_64 = sum([int(row[a]) for a in code_50_to_64])
            pop_65_over = sum([int(row[a]) for a in code_65_plus])
            pop_distribution_nonfamily = numpy.array([pop_18_49, pop_50_64, pop_65_over], dtype='f') / (pop_18_49 + pop_50_64 + pop_65_over)
            pop_distribution_family = numpy.array([pop_under_18, pop_18_49, pop_50_64, pop_65_over], dtype='f') / (pop_under_18 + pop_18_49 + pop_50_64 + pop_65_over)
            cbgs_to_age_distribution[check_str] = [pop_distribution_nonfamily, pop_distribution_family]

    print_elapsed_time()
    print('Reading social distancing data...')

    social_distancing_data = pd.read_csv(gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_social_distancing_metrics', WEEK[:4], WEEK[5:7], WEEK[8:], '{}-social-distancing.csv.gz'.format(WEEK)), 'rb'), error_bad_lines=False)
    cbgs_leaving_probs = {}  # probability that a member of a cbg will leave their house each tick
    seen_cbgs = set()
    for idx, row in social_distancing_data.iterrows():
        check_str = str(int(row['origin_census_block_group'])).zfill(12)
        if check_str in cbg_id_set:
            day_leaving_prob = PROPENSITY_TO_LEAVE * (1 - (int(row['completely_home_device_count']) / int(row['device_count'])))
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
    print('Caching raw data...')

    raw_cache_data = (cbg_ids, poi_set, hourly_lda, cbgs_to_age_distribution, cbgs_to_households, cbgs_leaving_probs, dwell_distributions, poi_type)
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

    agents = {}  # agent_id: [topic, infected_status, critical_date, home_cbg_code, age_code, household_id]
    households = {}  # household_id: {agent_id, agent_id, ...}
    cbgs_to_agents = {cbg: set() for cbg in cbg_ids}  # cbg: {agent_id, agent_id, ...}
    inactive_agent_ids = set()  # {agent_id, agent_id, ...}
    quarantined_agent_ids = {day: set() for day in range(SIMULATION_DAYS + QUARANTINE_DURATION)}  # {quarantine_expiry_date: {agent_id, agent_id, ...}, ...}
    active_agent_ids = {t: set() for t in range(total_simulation_time)}  # expiration_time: {(agent_id, poi_id), (agent_id, poi_id), ...}
    poi_current_visitors = {poi: set() for poi in poi_set}  # poi id: {agent_id, agent_id, ...} # may produce errors

    agent_count = 0
    household_count = 0


    def add_agent(current_cbg, household_id, possibly_child):
        global agent_count
        agent_id = 'agent_{}'.format(agent_count)
        agent_count += 1
        agent_topic = numpy.random.choice(topic_numbers, 1, p=probs)[0]
        if possibly_child:
            rand_age = numpy.random.choice(['C', 'Y', 'M', 'O'], p=cbgs_to_age_distribution[current_cbg][1])
        else:
            rand_age = numpy.random.choice(['Y', 'M', 'O'], p=cbgs_to_age_distribution[current_cbg][0])
        agent_status = 'S'
        rand_infected = 0
        permanently_quarantined = random.random() < ALPHA  # prohibits agent from ever leaving their house and ensures they are not initially infected if True
        if not permanently_quarantined:
            rand = random.random()
            if rand < PROPORTION_INITIALLY_INFECTED:
                rand /= PROPORTION_INITIALLY_INFECTED
                if rand < 0.23076923076:  # 2.754 / (2.754 + 1.928 + 2.662 + 4.590), from exposure gamma distributions
                    agent_status = 'E'
                    rand_infected = round(distribution_of_exposure.rvs(1)[0])
                elif rand < 0.53846153845:  # 40% of the remaining value, represents asymptomatic (subclinical) cases
                    agent_status = 'Ia'
                    rand_infected = round(distribution_of_subclinical.rvs(1)[0])
                elif rand < 0.7323278029:  # 1.928 / 4.590 of the remaining value, represents preclinical cases
                    agent_status = 'Ip'
                    rand_infected = round(distribution_of_preclinical.rvs(1)[0])
                else:  # represents symptomatic (clinical) cases
                    agent_status = 'Ic'
                    rand_infected = round(distribution_of_clinical.rvs(1)[0])
            inactive_agent_ids.add(agent_id)
        agents[agent_id] = [agent_topic, agent_status, rand_infected, current_cbg, rand_age, household_id]  # topic, infection status, critical date, cbg, age
        cbgs_to_agents[current_cbg].add(agent_id)
        households[household_id].add(agent_id)

    benchmark = 5
    for i, current_cbg in enumerate(cbg_ids):
        probs = numpy.array(hourly_lda[0][3][current_cbg]) # first hour, topic probabilities
        probs /= probs.sum()
        for idx, cbg_household_count in enumerate(cbgs_to_households[current_cbg]):
            current_household_size = idx % 7 + 1  # will end up being one more than this for family households
            for _ in range(cbg_household_count):
                household_id = 'household_{}'.format(household_count)
                household_count += 1
                households[household_id] = set()
                if idx >= 7:  # family households
                    add_agent(current_cbg, household_id, False)  # householder, cannot be a child
                    for _ in range(current_household_size):  # adds additional family members, can be children
                        add_agent(current_cbg, household_id, True)
                else:
                    for _ in range(current_household_size):
                        add_agent(current_cbg, household_id, False)
        completion = i / len(cbg_ids) * 100
        if completion >= benchmark:
            print('Preparing simulation... {}%'.format(adj_sig_figs(benchmark)))
            benchmark += 5

    print_elapsed_time()
    print('Caching agents data...')

    agents_cache_data = (agents, households, cbgs_to_agents, inactive_agent_ids, quarantined_agent_ids, active_agent_ids, poi_current_visitors)
    agents_cache_file = open(agents_cache_path, 'wb')
    pickle.dump(agents_cache_data, agents_cache_file)

    print_elapsed_time()
    agents_cache_file.close()

print('Number of agents: {}'.format(len(agents)))

print('Normalizing probabilities...')

for hour in range(168):
    aux_dict = {}  # an auxiliary POI dictionary for the least likely POIs per topic to speed up numpy.choice with many probabilities {topic: list of cbgs} {cbg: poi_ids, probabilities}
    for topic in range(len(hourly_lda[hour][2])):  # iterates through each topic
        minimum_list_index = -1
        prob_sum = 0
        while prob_sum < 0.2:  # the probability of selecting any POI in aux_dict is 20%
            prob_sum = sum(hourly_lda[hour][2][topic][1][minimum_list_index:])
            minimum_list_index -= 1
        aux_dict[topic] = [hourly_lda[hour][2][topic][0][minimum_list_index:], hourly_lda[hour][2][topic][1][minimum_list_index:]]  # [poi ids list, poi probabilities list for numpy.choice]
        aux_dict[topic][1] = numpy.array(aux_dict[topic][1])  # converts poi probabilities list at aux_dict[topic][1] from a list to a numpy array for numpy.choice
        aux_dict[topic][1] /= aux_dict[topic][1].sum()  # ensures the sum of aux_dict[topic][1] is 1 for numpy.choice
        # update old topics_to_pois with addition of aux, representing the selection of a POI from aux_dict
        hourly_lda[hour][2][topic][0] = hourly_lda[hour][2][topic][0][:minimum_list_index] + ['aux']
        hourly_lda[hour][2][topic][1] = numpy.array(hourly_lda[hour][2][topic][1][:minimum_list_index] + [prob_sum])
        hourly_lda[hour][2][topic][1] /= hourly_lda[hour][2][topic][1].sum()
    hourly_lda[hour].append(aux_dict)

print('Running simulation...')

# probability_of_infection = 0.005 / SIMULATION_TICKS_PER_HOUR  # from https://hzw77-demo.readthedocs.io/en/round2/simulator_modeling.html
age_susc = {'C': 0.01, 'Y': 0.025, 'M': 0.045, 'O': 0.085}  # "Susceptibility is defined as the probability of infection on contact with an infectious person" (https://www.nature.com/articles/s41591-020-0962-9)
asymptomatic_relative_infectiousness = 0.75  # https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
mask_reduction_factor = 0.35  # https://www.ucdavis.edu/coronavirus/news/your-mask-cuts-own-risk-65-percent/


def get_dwell_time(dwell_tuple):  # given a cached tuple from the dwell_distributions dictionary for a specific POI, return a dwell time in ticks
    dwell_time_minutes = 0  # represents dwell time in minutes (not ticks)
    while dwell_time_minutes < 30 / SIMULATION_TICKS_PER_HOUR:  # ensures dwell_time_ticks will be >= 1
        if len(dwell_tuple) == 3:
            dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(loc=dwell_tuple[1], scale=dwell_tuple[2], size=NUMBER_OF_DWELL_SAMPLES))
        elif len(dwell_tuple) == 4:
            dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(dwell_tuple[1], loc=dwell_tuple[2], scale=dwell_tuple[3], size=NUMBER_OF_DWELL_SAMPLES))
        else:
            dwell_time_minutes = statistics.median(getattr(scipy.stats, dwell_tuple[0]).rvs(dwell_tuple[1], dwell_tuple[2], loc=dwell_tuple[3], scale=dwell_tuple[4], size=NUMBER_OF_DWELL_SAMPLES))
    dwell_time_ticks = int(round(dwell_time_minutes * SIMULATION_TICKS_PER_HOUR / 60))  # represents dwell time in ticks
    return dwell_time_ticks


def infect(agent_id, day):  # infects an agent with the virus
    global infection_counts_by_day
    if agents[agent_id][1] == 'S':
        agents[agent_id][1] = 'E'
        agents[agent_id][2] = day + round(distribution_of_exposure.rvs(1)[0])
        infection_counts_by_day[day] += 1


def poi_infect(current_poi_agents, day, infectiousness):  # poi_agents, day, infectioness
    global agents
    for other_agent_id in current_poi_agents:
        rand = random.random()
        probability_of_infection = age_susc[agents[other_agent_id][4]] / SIMULATION_TICKS_PER_HOUR
        if rand < probability_of_infection * infectiousness:
            infect(other_agent_id, day)


def remove_expired_agents(t):  # removes agents from POIs whose visits end at time t
    global inactive_agent_ids, poi_current_visitors
    for tup in active_agent_ids[t]:
        inactive_agent_ids.add(tup[0])
        poi_current_visitors[tup[1]].remove(tup[0])
    active_agent_ids[t] = set()


def select_active_agents(t):  # removes agents from POIs whose visits starting at at time t
    global inactive_agent_ids, active_agent_ids, poi_current_visitors
    to_remove = set()
    for agent_id in inactive_agent_ids:
        if random.random() < cbgs_leaving_probs[agents[agent_id][3]]:
            topic = agents[agent_id][0]
            destination = numpy.random.choice(hourly_lda[t][3][topic][0], 1, p=hourly_lda[t][3][topic][1])[0]
            if destination == 'aux':
                destination = numpy.random.choice(hourly_lda[t][4][topic][0], 1, p=hourly_lda[t][4][topic][1])[0]
            '''
            rerolls = 0
            while rerolls < MAXIMUM_REROLLS and destination in closed_pois:
                destination = numpy.random.choice(topics_to_pois[topic][0], 1, p=topics_to_pois[topic][1])[0]
                if destination == 'aux':
                    destination = numpy.random.choice(aux_dict[topic][0], 1, p=aux_dict[topic][1])[0]
                destination_end_time = t + get_dwell_time(dwell_distributions[destination])
                rerolls += 1
            if rerolls == MAXIMUM_REROLLS:
                continue
            '''
            destination_end_time = t + get_dwell_time(dwell_distributions[destination])
            if destination in closed_pois:
                continue
            if destination_end_time >= total_simulation_time:
                continue
            active_agent_ids[destination_end_time].add((agent_id, destination))
            poi_current_visitors[destination].add(agent_id)
            to_remove.add(agent_id)
    inactive_agent_ids -= to_remove


def reset_all_agents():  # forces all agents to return home from POIs
    global inactive_agent_ids, active_agent_ids, poi_current_visitors
    for t in active_agent_ids:
        for tup in active_agent_ids[t]:
            inactive_agent_ids.add(tup[0])
            poi_current_visitors[tup[1]].remove(tup[0])
        active_agent_ids[t] = set()


def household_transmission(day):  # simulates a day's worth of of household transmission in every household
    for household in households.values():
        base_transmission_probability = .091/4.59 if len(household) < 6 else .204/4.59  # https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30471-0/fulltext
        for agent_id in household:
            if agents[agent_id][1] == 'Ia' or agents[agent_id][1] == 'Ip':
                for other_agent_id in household:
                    if random.random() < base_transmission_probability * asymptomatic_relative_infectiousness:
                        infect(other_agent_id, day)
            elif agents[agent_id][1] == 'Ic':
                for other_agent_id in household:
                    if random.random() < base_transmission_probability:
                        infect(other_agent_id, day)


def set_agent_flags(day):  # changes agent infection status
    global agents
    for key in agents:
        if agents[key][1] == 'E' and agents[key][2] == day + 1:  # https://www.nature.com/articles/s41586-020-2196-x?fbclid=IwAR1voT8K7fAlVq39RPINfrT-qTc_N4XI01fRp09-agM1v5tfXrC6NOy8-0c
            rand_s = random.random()
            rand = 0
            if rand_s < 0.4:  # subclinical (asymptomatic) cases, never show symptoms
                agents[key][1] = 'Ia'
                rand = round(distribution_of_subclinical.rvs(1)[0])
            else:  # preclinical cases, will be symptomatic in the future
                agents[key][1] = 'Ip'
                rand = round(distribution_of_preclinical.rvs(1)[0])
            agents[key][2] = day + rand
        elif agents[key][1] == 'Ip' and agents[key][2] == day + 1:
            agents[key][1] = 'Ic'
            agents[key][2] = round(distribution_of_clinical.rvs(1)[0])
        elif (agents[key][1] == 'Ia' or agents[key][1] == 'Ic') and agents[key][2] == day + 1:
            agents[key][1] = 'R'


def report_status():  # prints status
    global infection_counts_by_day
    stat = {'S': 0, 'E': 0, 'Ia': 0, 'Ip': 0, 'Ic': 0, 'R': 0, 'Q': 0}
    for tup in agents.values():
        stat[tup[1]] += 1
    print('Susceptible: {}'.format(stat['S']))
    print('Exposed: {}'.format(stat['E']))
    print('Infected: {}'.format(stat['Ia']+stat['Ip']+stat['Ic']))
    print('Recovered: {}'.format(stat['R']))
    print('Quarantined: {}'.format(len(quarantined_agent_ids)))


def quarantine(ag, days, d):  # agents, number of days in quarantine, current day => agents cannot leave house during quarantine
    global quarantined_agent_ids, inactive_agent_ids
    for a in ag:
        quarantined_agent_ids[d + days].add(a)
        inactive_agent_ids.remove(a)
    for q in list(quarantined_agent_ids[d]):
        quarantined_agent_ids[d].remove(q)
        inactive_agent_ids.add(q)


def wear_masks():  # simulates the entire population wearing masks
    global age_susc
    for a in age_susc:
        age_susc[a] *= mask_reduction_factor


def close_poi_type(current_poi_type):  # e.g. "Restaurants and Other Eating Places"
    global closed_pois
    closed_pois |= poi_type[current_poi_type]


# Infected status
# S => Susceptible: never infected
# E => Exposed: infected the same day, will be contagious after incubation period
# Infected: 60% of cases are Ip (presymptomatic) then Ic (symptomatic), 40% of cases are Ia (asymptomatic, these agents never show symptoms)
#     Ip => Preclinical, before clinical, not symptomatic, contagious
#     Ic => Clinical, full symptoms, contagious
#     Ia => Asymptomatic (subclinical), 75% relative infectioness, never show symptoms
# R => Recovered: immune to the virus, no longer contagious

# prescriptions
wear_masks()
closed_pois = set()
for current_poi_type in CLOSED_POI_TYPES:
    close_poi_type(current_poi_type)

infection_counts_by_day = []
for day in range(SIMULATION_DAYS):
    print('Day {}:'.format(day))
    infection_counts_by_day.append(0)
    to_be_quarantined = set()
    for current_time in range(total_simulation_time):
        remove_expired_agents(current_time)
        if current_time != total_simulation_time - 1:
            for poi in poi_current_visitors:
                if poi not in closed_pois:
                    if len(poi_current_visitors[poi]) > 1:
                        poi_agents = list(poi_current_visitors[poi])
                        max_interactions = min(len(poi_agents) - 1, MAXIMUM_INTERACTIONS_PER_TICK)
                        for agent_idx, agent_id in enumerate(poi_agents):
                            if agents[agent_id][1] == 'Ia' or agents[agent_id][1] == 'Ip':
                                possible_agents = poi_agents[:agent_idx] + poi_agents[agent_idx+1:]
                                poi_infect(np.random.choice(possible_agents, max_interactions), day, asymptomatic_relative_infectiousness)
                            elif agents[agent_id][1] == 'Ic':
                                possible_agents = poi_agents[:agent_idx] + poi_agents[agent_idx+1:]
                                poi_infect(np.random.choice(possible_agents, max_interactions), day, 1)
                                to_be_quarantined.add(agent_id)  # quarantines symptomatic agents one day after they begin showing symptoms
            select_active_agents(current_time)

        if not current_time % SIMULATION_TICKS_PER_HOUR:
            print('Hour {} complete.'.format(int(current_time / SIMULATION_TICKS_PER_HOUR)))
            print_elapsed_time()

    household_transmission(day)
    print('Daily household transmission complete.')
    print_elapsed_time()

    reset_all_agents()
    quarantine(to_be_quarantined, QUARANTINE_DURATION, day)
    print('Day {} complete. Infection status:'.format(day))
    report_status()
    print('New cases per day: {}'.format(list(zip([i for i in range(len(infection_counts_by_day))], infection_counts_by_day))))
    print()

    set_agent_flags(day)

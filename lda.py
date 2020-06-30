import pandas as pd
import gensim
import json
import random
import os

MINIMUM_RAW_VISITOR_COUNT = 25


def lda(area, week):
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(week), '{}-{}.csv'.format(area, week)), error_bad_lines=False)
    usable_data = data[(data.raw_visitor_counts >= MINIMUM_RAW_VISITOR_COUNT) & (data.visitor_home_cbgs != '{}')]

    lda_documents = []  # [['cbg_510594504002', 'cbg_510594808011', ... 'cbg_511076112062'], ['cbg_510594808011', 'cbg_510594808011', ... 'cbg_511076112062'], ...]
    place_ids = []
    place_counts = []

    for row in usable_data.itertuples():
        place_ids.append(str(row.safegraph_place_id))
        cbgs = json.loads(row.visitor_home_cbgs)
        lda_words = []  # ['cbg_510594504002',  ... 'cbg_511076112062', 'cbg_511076112062']
        total_frequency = 0
        for cbg in cbgs:
            cbg_frequency = random.randint(2, 4) if cbgs[cbg] == 4 else cbgs[cbg]
            total_frequency += cbg_frequency
            lda_words.extend([cbg] * cbg_frequency)
        place_counts.append(total_frequency)
        lda_documents.append(lda_words)

    lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents)
    lda_corpus = [lda_dictionary.doc2bow(place) for place in lda_documents]
    place_to_bow = dict(zip(place_ids, lda_corpus))
    place_to_counts = dict(zip(place_ids, place_counts))
    # id_to_cbg = dict(lda_dictionary.token2id)
    lda_model = gensim.models.HdpModel(lda_corpus, id2word=lda_dictionary)
    # lda_model = gensim.models.LdaModel(lda_corpus, num_topics=NUM_TOPICS, id2word=lda_dictionary)
    return lda_model, lda_corpus, place_to_bow, place_to_counts
    

def main():
    from mapper import cbgs_to_map

    area = input('Locality (examples: fairfax-county-va (default) or state-nm): ')
    if area == '':
        area = 'fairfax-county-va'
    week = input('Week (default: 2020-06-01): ')
    if week == '':
        week = '2020-06-01'
    
    lda_model, lda_corpus, place_to_bow, place_to_counts = lda(area, week)
    print('Complete! Displaying visualization in web browser...')
    print()
    num_topics = len(lda_model.get_topics())
    top_topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
    cbgs_to_map([topic[1] for topic in top_topics], ['topic_{}'.format(idx) for idx in range(len(top_topics))])

    topic_to_places = [list() for i in range(num_topics)]
    for key in place_to_bow:
        for tup in lda_model[place_to_bow[key]]:
            topic_to_places[tup[0]].append((tup[1] * place_to_counts[key], key))
    for idx, elem in enumerate(topic_to_places):
        total_weights = 0
        for tup in elem:
            total_weights += tup[0]
        topic_to_places[idx] = sorted([(tup[0] / total_weights, tup[1]) for tup in elem], reverse=True)

    while True:
        input_str = str(input('Enter SafeGraph place ID or topic number: '))
        if input_str[:3] == 'sg:':
            if input_str in place_to_bow:
                print(lda_model[place_to_bow[input_str]])
            else:
                print('Not found.')
        else:
            print(topic_to_places[int(input_str)])
        print()


if __name__ == "__main__":
    main()

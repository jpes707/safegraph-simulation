import pandas as pd
import gensim
import json
import random
import os

MINIMUM_RAW_VISITOR_COUNT = 25
NUM_TOPICS = 15


def lda(area, week):
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safegraph-data', 'safegraph_weekly_patterns_v2', 'main-file', '{}-weekly-patterns'.format(week), '{}-{}.csv'.format(area, week)), error_bad_lines=False)
    usable_data = data[(data.raw_visitor_counts >= MINIMUM_RAW_VISITOR_COUNT) & (data.visitor_home_cbgs != '{}')]

    lda_documents = []  # [['cbg_510594504002', 'cbg_510594808011', ... 'cbg_511076112062'], ['cbg_510594808011', 'cbg_510594808011', ... 'cbg_511076112062'], ...]

    for row in usable_data.head().itertuples():
        cbgs = json.loads(row.visitor_home_cbgs)
        lda_words = []  # ['cbg_510594504002',  ... 'cbg_511076112062', 'cbg_511076112062']
        for cbg in cbgs:
            cbg_frequency = cbgs[cbg]
            if cbg_frequency == 4:
                cbg_frequency = random.randint(2, 4)
            lda_words.extend([cbg] * cbg_frequency)  # 'cbg_' + 
        lda_documents.append(lda_words)

    lda_dictionary = gensim.corpora.dictionary.Dictionary(lda_documents)
    lda_corpus = [lda_dictionary.doc2bow(place) for place in lda_documents]
    # id_to_cbg = dict(lda_dictionary.token2id)
    lda_model = gensim.models.HdpModel(lda_corpus, id2word=lda_dictionary)
    return lda_model, lda_corpus
    

def main():
    from mapper import cbgs_to_map

    area = input('Locality (examples: fairfax-county-va or state-nm): ')
    if area == '':
        area = 'fairfax-county-va'
    week = input('Week (example: 2020-05-25): ')
    if week == '':
        week = '2020-06-01'
    
    lda_model, lda_corpus = lda(area, week)
    print('Complete! Displaying visualization in web browser...')
    top_topics = lda_model.show_topics(num_topics=NUM_TOPICS, formatted=False)
    cbgs_to_map([topic[1] for topic in top_topics], ['topic_{}'.format(idx) for idx in range(len(top_topics))])


if __name__ == "__main__":
    main()

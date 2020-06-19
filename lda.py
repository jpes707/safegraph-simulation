import pandas as pd
import gensim
import json
import random
from mapper import *

AREA = 'state-ri'  # state-nm or fairfax-county-va
WEEK = '2020-06-01'
MINIMUM_RAW_VISITOR_COUNT = 25

data = pd.read_csv(r'E:\safegraph-simulation\safegraph-data\safegraph_weekly_patterns_v2\main-file\{}-weekly-patterns.csv\{}-{}.csv'.format(WEEK, AREA, WEEK), error_bad_lines=False)
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
id_to_cbg = dict(lda_dictionary.token2id)

lda = gensim.models.LdaModel(lda_corpus, num_topics=15, id2word=lda_dictionary, alpha='auto', eval_every=5)
print()
print()
print()
print(lda.top_topics(lda_corpus))
lda_to_map(lda.top_topics(lda_corpus))

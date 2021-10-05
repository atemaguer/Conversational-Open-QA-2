import os
import json
import csv

import pandas as pd
import numpy as np
import ujson
import spacy

train_passages_path = "../data/QRECC/corpus.tsv"
train_corpus_path = "../data/QRECC/train/corpus.tsv"
dev_queries_path = "../data/QRECC/test/queries.tsv"
qrecc_path = "../data/qrecc_data/train.json"
queries_positive_passages = "../data/QRECC/query_positives.tsv"

nlp = spacy.load("en_core_web_lg")


def jsonl_to_list(path):

    with open(path) as f:
        return [ujson.loads(line) for line in f]

def extract_most_similar_passage(passage, df):
    doc = nlp(passage)
    df = df.assign(sim_score = df['PASSAGE'].apply(lambda x: 0  if len(x) == 0 else nlp(str(x)).similarity(doc)))
    return df.loc[df.sim_score.idxmax()]['PASSAGE']


qrecc_df = pd.DataFrame(json.load(open(qrecc_path)))

passages_df = pd.read_csv(train_passages_path, sep="\t", names=["PASSAGE","PAGE_URL"], index_col=False)


def generate_positive_examples():
    with open(queries_positive_passages, 'w') as outfile:

        writer = csv.writer(outfile, delimiter='\t')

        for idx, key in enumerate(qrecc_df.index):
            context = qrecc_df.loc[key, 'Context']
            question = qrecc_df.loc[key, 'Question']
            answer = qrecc_df.loc[key, 'Answer']
            answer_url = qrecc_df.loc[key, 'Answer_URL']
            conversation_no = qrecc_df.loc[key, 'Conversation_no']
            turn_no = qrecc_df.loc[key, 'Turn_no']

            query = [question]

            if len(answer) == 0:
                continue

            if len(context) == 1:
                query = [question, context[-1]]
            elif len(context) == 2:
                query = [question, context[-2], context[-1]]
            elif len(context) == 3:
                query = [question, context[-3], context[-1], context[-2]]



            similar_passages_df = passages_df[passages_df['PAGE_URL'] == answer_url]

            if len(similar_passages_df) == 0:
                continue

            positive_passage = extract_most_similar_passage(answer, similar_passages_df)

            writer.writerow([" | ".join(query), positive_passage, answer, answer_url])

            if idx % 1000 == 0:
                print(f'Processed {idx} passages')


def generate_negative_examples():
    positive_passages_df = pd.read_csv(queries_positive_passages, index_col=False, sep="\t", names=["query","positive_passage","passage_url"])

    positive_passages_df = positive_passages_df.sample(frac = 1)
    
    num_pos_passages = len(positive_passages_df) #shuffle the passages to ensure randomness
    num_dev = (num_pos_passages) // 10 #10%
    
    dev_df = positive_passages_df.iloc[:num_dev-1, :]
    train_df = positive_passages_df.iloc[num_dev:, :]

    print(f'{num_pos_passages} positives, {len(dev_df)} dev, {len(train_df)} train')

#     with open(train_corpus_path, "w") as outfile:
#         df = pd.DataFrame()

#         for idx, key in enumerate(train_df.index):
#             passage_url = train_df.loc[key, 'passage_url']
#             query = train_df.loc[key, 'query']
#             positive_passage = train_df.loc[key, 'positive_passage']
#             negative_passages_df = passages_df[positive_passages_df["PAGE_URL"] == passage_url]

#             negative_passages = negative_passages_df.sample(10, random_state=10)

#             for passage_idx in negative_passages.index:
#                 negative_passage = negative_passages.loc[passage_idx, "passage"]

#             if idx == 2:
#                 break

#         print(f'{len(df)} passages')

#         df.to_csv(train_corpus_path)
#      dev_df.to_csv()

if __name__ == "__main__":
    generate_negative_examples()
    #generate_positive_examples()

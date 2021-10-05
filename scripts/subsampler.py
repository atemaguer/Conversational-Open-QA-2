import os
import json
import re
import copy
import operator
import itertools
import random
import csv

import pandas as pd
import numpy as np
import ujson

from collections import defaultdict
from functools import reduce

WAYBACK_DATASET = "../data/wayback.jsonl"
QRECC_DATASET = "../data/qrecc_data/train.json"
QUERIES_SAVE_PATH = "../data/subsample/queries.tsv"
CORPUS_SAVE_PATH = "../data/subsample/corpus.tsv"
NEGATIVE_PATHS = ["../data/QRECC/wayback/0.jsonl", "../data/QRECC/wayback/1.jsonl","../data/QRECC/wayback-backfill/0.jsonl"]
COMMON_CRAWL_ROOT = "../data/QRECC/commoncrawl"

NEGATIVE_PATHS += [f'../data/QRECC/commoncrawl/{dirent}' for dirent in os.listdir(COMMON_CRAWL_ROOT)]

NEGATIVE_PATHS.append("../data/wayback.jsonl")

COMBINED_CORPUS_SAVE_PATH = "../data/QRECC/corpus.tsv"

def from_jsonl_to_list(path):
    with open(path) as f:
        return [ujson.loads(line) for line in f]

def extract_url(passage_id):
    rest = passage_id.split("_")[1:]
    rest = "".join(rest[:-1])
    url = rest.lstrip("/")
    return url

def map_url_to_passages(df):
    url_to_passages = defaultdict(list)
    for idx in range(len(df)):
        passage = df.loc[idx]
        url = extract_url(passage['id'])
        url_to_passages[url].append(passage['contents'])

    return url_to_passages

def extract_gold_and_negative_passages(questions_df, url_to_passages):
    passages = copy.deepcopy(url_to_passages)
    gold_passages = {}

    for idx in questions_df.index:
        answer_url = questions_df.loc[idx,'Answer_URL']
        if answer_url in passages:
            gold_passages[f'{answer_url}'] = passages[answer_url]
            del passages[answer_url]

    return gold_passages, passages


def generate_subsample_dataset(num_questions, num_neg_passages, context_size=5):
    wayback_df = pd.DataFrame(from_jsonl_to_list(WAYBACK_DATASET))
    qrecc_df = pd.DataFrame(json.load(open(QRECC_DATASET)))
    qrecc_df = qrecc_df.assign(Context_size = qrecc_df['Context'].apply(lambda x: len(x)))

    url_to_passages = map_url_to_passages(wayback_df)

    answerable_df = qrecc_df[qrecc_df['Answer_URL'].isin(url_to_passages.keys())]

    subsample_df = answerable_df[answerable_df['Context_size'] > 8].sample(num_questions, random_state=10)

    gold_passages, negative_passages = extract_gold_and_negative_passages(subsample_df, url_to_passages)

    neg_passages_subsample = list(negative_passages.items())  #random.sample(list(negative_passages.items()), num_neg_passages)

    subsample_corpus = list(gold_passages.items()) + neg_passages_subsample

    num_passages = 0

    with open(CORPUS_SAVE_PATH, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for idx, example in enumerate(subsample_corpus):
            url = example[0]
            passages = example[1]
            for passage in passages:
                if len(passage) > 1:
                    writer.writerow(['id', passage, 'page title', url])
                    num_passages += 1

         #add negative passages
        for path in NEGATIVE_PATHS:
            df = pd.DataFrame(from_jsonl_to_list(os.path.join(ROOT_NEGATIVES_PATH, path)))

            for idx in df.index:
                passage = df.loc[idx, "contents"]
                url = extract_url(df.loc[idx, "id"])
                if len(passage) > 1:
                    num_passages += 1
                    writer.writerow(['id', passage, 'page title', url])

    with open(QUERIES_SAVE_PATH, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for idx in subsample_df.index:
            context = subsample_df.loc[idx, 'Context']
            question = subsample_df.loc[idx, 'Question']
            answer = subsample_df.loc[idx, 'Answer']
            answer_url = subsample_df.loc[idx, 'Answer_URL']
            conversation_no = subsample_df.loc[idx, 'Conversation_no']
            turn_no = subsample_df.loc[idx, 'Turn_no']
            query = " | ".join([question,context[0], context[1], context[-2]])
            writer.writerow([idx, query, answer, answer_url, conversation_no, turn_no])

    print(f'The corpus contains {num_passages} passages')


def generate_training_corpus():
    num_passages = 0

    with open(COMBINED_CORPUS_SAVE_PATH, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')

        for path in NEGATIVE_PATHS:
            df = pd.DataFrame(from_jsonl_to_list(path))

            for idx in df.index:
                passage = df.loc[idx, "contents"]
                url = extract_url(df.loc[idx, "id"])
                if len(passage) > 1:
                    num_passages += 1
                    writer.writerow([passage, url])

    print(f'The corpus contains {num_passages} passages')

if __name__ == "__main__":
    generate_training_corpus()
#     generate_subsample_dataset(300, 3000)


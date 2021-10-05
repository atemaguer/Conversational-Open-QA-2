import os
import json
import csv

import pandas as pd
import numpy as np

CORPUS_PATH = "../data/subsample/corpus.tsv"
QUERIES_PATH = "../data/subsample/queries.tsv"
RANKINGS_PATH = "/home/ubuntu/Conversational-QA/experiments/QRECC/retrieve.py/2021-08-10_05.07.35/ranking.tsv"

rankings_df = pd.read_csv(RANKINGS_PATH, sep='\t', header=None, names=['Qid', 'Answer_ID', 'Rank'])

queries_df = pd.read_csv(QUERIES_PATH, sep='\t', header=None, names=['Question', 'Answer', 'Answer_URL', 'Conversation_no', 'Turn_no'], index_col=0)

corpus_df = pd.read_csv(CORPUS_PATH, sep='\t', header=None, names=['ID', 'Content', 'Page_Title', 'Page_URL'])

def success_at_k(k=5):
    success = 0
    num_queries = queries_df.shape[0]

    for idx in queries_df.index:
        query_answer_url = queries_df.loc[idx, 'Answer_URL']
        passages = rankings_df[rankings_df['Qid'] == idx]
        answers = corpus_df.loc[passages['Answer_ID']]
        page_urls = list(answers['Page_URL'])

        if query_answer_url in page_urls[:k]:
            success += 1

    print(f'Success @ {k} is {success/num_queries}!')

def main():
    for k in [1,5,10,20,100]:
        success_at_k(k)

if __name__ == '__main__':
    main()



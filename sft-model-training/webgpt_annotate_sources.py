import asyncio
from collections import defaultdict
from datasets import load_dataset
import json
import requests
import jsonlines
import re
from tqdm import tqdm
import numpy as np

def get_snippets_from_query(query, keys):
    api_key = keys['google_custom_search']
    search_engine_id = keys['search_engine_id']
    payload = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query
    }
    req = requests.get("https://customsearch.googleapis.com/customsearch/v1", payload)
    try:
        items = req.json()['items'][:20]
    except KeyError:
        return None
    snippets = []
    for item in items:
        try:
            snippets.append(item['snippet'])
        except KeyError:
            continue
    return snippets[:5]
    

def main():
    with open("api_keys.json", 'r') as f:
        keys = json.load(f)

    with jsonlines.open("kcc_data/jsonlines/webgpt.jsonl", mode = 'w') as f:
        webgpt_data = load_dataset("openai/webgpt_comparisons", cache_dir = 'hf-data', split = 'train')
        for record in tqdm(webgpt_data):
            if record['score_1'] < 0 and record['score_0'] < 0:
                continue
            query = record['question']['full_text']
            snippets = get_snippets_from_query(query, keys)
            if snippets is None:
                continue
            data = {
                'query': query,
                'snippets': snippets
            }
            if record['score_1'] > record['score_0']:
                true_answer = record['answer_1']
            else:
                true_answer = record['answer_0'] 

            true_answer = re.sub("\[([0-9])*\]", "", true_answer)
            true_answer = true_answer.replace("Bing", "Raithubot")
            true_answer = true_answer.replace("bing", "bing")
            data['true_answer'] = true_answer
            f.write(data)

if __name__ == '__main__':
    main()
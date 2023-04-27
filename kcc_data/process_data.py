from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import numpy as np
import torch
import argparse

def generate(text, tokenizer, model):
    tokens = torch.cuda.IntTensor(tokenizer([text]).input_ids)
    generation = model.generate(
        input_ids = tokens,
        temperature = 0.0,
        top_k = 0,
        top_p = 0,
        max_length = 512
    )
    text = tokenizer.decode(generation[0].cpu())
    text = text.split("\n")
    prompt = text[13][len("Question: "):]
    summary = text[14][len("Summary: "):]
    return prompt, summary

def isvalid(query):
    if "number" in query.lower():
        return False
    if "weather" in query.lower():
        return False
    
    return True

def read_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    np.random.shuffle(data)
    for record in data:
        if isvalid(record['QueryText']):
            yield record["QueryText"]


def read_available_data():
    for root, dirs, files in os.walk("kcc_data/json/"):
        for file in files:
            yield from read_from_json(os.path.join(root,file))


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(
        'huggyllama/llama-13b', 
        cache_dir = 'hf-models/',
    ).half().eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-13b', cache_dir = 'hf-models/')

    input_prompt = (
        "Prompt: varieties of maize\n"
        "Question: I am new to farming. Could you tell me the various types of maize that can be grown in telangana?\n"
        "Summary: Types of maize crops grown in telangana\n\n"
        "Prompt: stem borer management in paddy\n"
        "Question: I'm having this moth in my paddy farm that are pale yellow in color with a dark brown head. I'm not sure what to do to remove them. Can you help me?\n"
        "Summary: Stem borer management in paddy\n\n"
        "Prompt: FARMER ASKED QUERY ON NUTRIENT MANAGEMENT IN BENGAL GRAM\n"
        "Question: I'm growing bengal gram in my farmland. But I'm not sure if it's having appropriate nutrients. Could you help me figure it out?\n"
        "Summary: Nutrient Management in bengal gram\n\n"
        "Prompt: {}\n"
        "Question: "
        ""
    )

    writer = open("kcc_data/text/data.txt", 'w')

    for query in tqdm(read_available_data()):
        prompt = input_prompt.format(query.strip())
        try:
            response, summary = generate(prompt, tokenizer, model)
            writer.write(response + "\n")
            writer.write(summary + "\n")
            writer.flush()
        except IndexError as e:
            print(e)
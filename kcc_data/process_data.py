from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import torch

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

    text = text[text.rfind("<|assistant|>"):text.rfind("<|endoftext|>")]

    text = text[len("<|assistant|>"):]

    text = text[text.find("\"")+1:text.rfind("\"")]
    return text


def read_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    for record in data:
        yield record["QueryText"]


def read_available_data():
    for root, dirs, files in os.walk("kcc_data/json/"):
        for file in files:
            yield from read_from_json(os.path.join(root,file))


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained('OpenAssistant/oasst-sft-1-pythia-12b', cache_dir = 'hf-models/')
    model = model.half().eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained('OpenAssistant/oasst-sft-1-pythia-12b', cache_dir = 'hf-models/')

    input_prompt = ("<|prompter|>Rephrase the prompt from a confused farmer:"
     " \'inquiering date to distributed the new land use policy NLUP of the mizoram state goverment.\'"
     "<|endoftext|><|assistant|>Rephrased prompt: \"Hey There I am a farmer from mizoram and I'd like to know when will the new land use policy\""
     " NLUP will start?<|endoftext|><|prompter|>Rephrase the text from a confused farmer:"
     "\'INFORMATION OF GRAO VINE\'<|endoftext|><|assistant|>Rephrased prompt: \"So I heard about grao vine from my friends and I am not able to understand if it's useful to me. What is Grao Vine? And what are it's uses?\""
     "<|endoftext|>"
     "<|prompter|>Rephrase the text from a confused farmer:"
     "\'{}\'<|endoftext|><|assistant|>Rephrased Prompt: \""
    )

    writer = open("kcc_data/text/data.txt", 'w')

    for query in tqdm(read_available_data()):
        prompt = input_prompt.format(query)
        
        response = generate(prompt, tokenizer, model)
        writer.write(response + "\n")

        writer.flush()



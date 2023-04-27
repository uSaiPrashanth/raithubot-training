from transformers import AutoModelForCausalLM, AutoTokenizer
# from oslo import ParallelContext, ParallelMode
# from oslo.torch.nn.parallel import TensorParallel
# from oslo.torch.utils.extensions import from_parallelized
from functools import partial
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader, IterableDataset
import oslo
import os
from tqdm.auto import tqdm
import numpy as np
import jsonlines
import torch

class WebDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.data = data
        
        self.tokenizer = tokenizer
        np.random.shuffle(self.data)
    
    def __getitem__(self, idx):
        query = self.data[idx]['query']
        snippets = self.data[idx]['snippets']
        response = self.data[idx]['true_answer']
        
        train_seq = "Query: " + query + "\n"
        train_seq += "Response: " + response + "<|endoftext|>"
        ids = self.tokenizer(
            train_seq, 
            max_length = 2048, 
            return_tensors = 'pt',
            truncation = True,
        )
        return {
            'input_ids': ids['input_ids']
        }
    
    def __iter__(self):
        while True:
            for i in range(len(self)):
                yield self[i]
            np.random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)

def train_model(parallel_context):
    model = AutoModelForCausalLM.from_pretrained(
        'EleutherAI/pythia-2.8b', 
        cache_dir = './hf-models'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'EleutherAI/pythia-2.8b', 
        cache_dir = './hf-models',
    )
    tokenizer.pad_token = tokenizer.eos_token

    with jsonlines.open("kcc_data/jsonlines/kcc.jsonl", mode = 'r') as reader:
        data = []
        for obj in reader:
            if not obj['snippets'] is None:
                data.append(obj)

    # model = TensorParallel(model, parallel_context)
    # oslo.ready(model, parallel_context)
    

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config).cuda()
    
    train_data = iter(WebDataset(tokenizer, data[:-10]))
    test_data = iter(WebDataset(tokenizer, data[-10:]))
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    gradient_accumilation_steps = 8
    pbar = tqdm(range(250))
    for _ in pbar:
        step = 0
        tokens = next(train_data)
        optimizer.zero_grad()
        input_ids = tokens['input_ids'][:, 0:-1].cuda()
        labels = tokens['input_ids'][0, 1:].cuda()

        with torch.cuda.amp.autocast():
            outputs = model.forward(input_ids = input_ids).logits[0]
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumilation_steps == 0:
            scaler.step(optimizer)
            lr_scheduler.step()
            scaler.update()

            avg_loss = 0
            with torch.no_grad():
                for _ in range(10):
                    tokens = next(test_data)
                    input_ids = tokens['input_ids'][:, 0:-1].cuda()
                    labels = tokens['input_ids'][0, 1:].cuda()

                    with torch.cuda.amp.autocast():
                        outputs = model.forward(input_ids = input_ids).logits[0]
                        avg_loss += loss_fn(outputs, labels).item()
                
                pbar.set_description(f"Loss: {loss.item():.3f} Validation Loss: {loss.item():.3f}")
            step += 1

    model.save_pretrained(save_directory = "./hf-models/usvsnsp-sft-model/")
    tokenizer.save_pretrained(save_directory = "./hf-models/usvsnsp-sft-model/")
    tokenizer.save_pretrained(save_directory = "./hf-models/usvsnsp-sft-combined/")

def deparallelize_save(parallel_context):
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-6.9b', cache_dir = './hf-models')
    model = TensorParallel(model, parallel_context)
    oslo.ready(model, parallel_context)

    model.from_parallelized(path = "./hf-models/usvsnsp-sft-model")
    model.save_pretrained(
        save_directory = './hf-models/usvsnsp-sft-combined',
        merge_checkpoints = True
    )

def main():
    # parallel_context = ParallelContext.from_torch(
    #     data_parallel_size=1,
    #     pipeline_parallel_size=1,
    #     tensor_parallel_size=int(os.environ['WORLD_SIZE']),
    #     tensor_parallel_mode=ParallelMode.TENSOR_1D,
    # )
    train_model(None)
    torch.cuda.empty_cache()
    deparallelize_save(parallel_context)

if __name__ == '__main__':
    main()
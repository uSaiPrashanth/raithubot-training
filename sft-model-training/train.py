from transformers import AutoModelForCausalLM, AutoTokenizer
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import TensorParallel
from oslo.torch.utils.extensions import from_parallelized
from functools import partial
import oslo
import os
from tqdm.auto import tqdm
import jsonlines
import torch


def train_model(parallel_context):
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-12b', cache_dir = './hf-models')
    tokenizer = AutoTokenizer.from_pretrained(
        'EleutherAI/pythia-12b', 
        cache_dir = './hf-models',
    )

    with open("kcc_data/jsonlines/data.jsonl", 'r') as reader:
        data = [i[1:-2] for i in reader]

    model = TensorParallel(model, parallel_context)
    oslo.ready(model, parallel_context)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)

    with tqdm(data) as pbar:
        for record in pbar:
            tokens = tokenizer(
                [record], 
                return_tensors = 'pt',
                truncation = True,
                max_length = 2048
            )
            input_ids = tokens['input_ids'][:, 0:-1].cuda()
            labels = tokens['input_ids'][:, 1:].cuda()
    
            optimizer.zero_grad()
            loss = model(input_ids = input_ids, labels = labels).loss
            loss.backward()
            optimizer.step()

            pbar.set_description(f"LOSS: {loss.item():.3f}")
        
        
    model.save_pretrained(save_directory = "./hf-models/usvsnsp-sft-model/")

def deparallelize_save(parallel_context):
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-12b', cache_dir = './hf-models')
    model = TensorParallel(model, parallel_context)
    oslo.ready(model, parallel_context)

    model.from_parallelized(path = "./hf-models/usvsnsp-sft-model")
    model.save_pretrained(
        save_directory = './hf-models/usvsnsp-sft-combined',
        merge_checkpoints = True
    )

if __name__ == '__main__':
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=int(os.environ['WORLD_SIZE']),
        tensor_parallel_mode=ParallelMode.TENSOR_1D,
    )
    # train_model(parallel_context)
    deparallelize_save(parallel_context)

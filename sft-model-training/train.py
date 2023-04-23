from transformers import AutoModelForCausalLM, AutoTokenizer
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import TensorParallel
from oslo.torch.utils.extensions import from_parallelized
from functools import partial
import oslo
import os
from tqdm.auto import tqdm
import numpy as np
import jsonlines
import torch

def train_model(parallel_context):
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neox-20b', cache_dir = './hf-models')
    tokenizer = AutoTokenizer.from_pretrained(
        'EleutherAI/gpt-neox-20b', 
        cache_dir = './hf-models',
        additional_special_tokens = (
            '<|query|>',
            '</query/>',
            '<|response|>',
            '</response/>',
            '<|result|>',
            '</result/>',
            '<|summary|>',
            '</summary/>',
            '<|results|>',
            '</results/>',
            '<|result-sentence|>'
        )
    )

    with open("kcc_data/jsonlines/trimmed.jsonl", 'r') as reader:
        data = [i[1:-2] for i in reader]

    model = TensorParallel(model, parallel_context)
    oslo.ready(model, parallel_context)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)

    gradient_accumilation_steps = 8
    for epoch in range(20):
        np.random.shuffle(data)
        with tqdm(data) as pbar:
            step = 0
            for record in pbar:
                optimizer.zero_grad()
                tokens = tokenizer(
                    [record], 
                    return_tensors = 'pt',
                    truncation = True,
                    max_length = 1536
                )
                input_ids = tokens['input_ids'][:, 0:-1].cuda()
                labels = tokens['input_ids'][:, 1:].cuda()

                with torch.cuda.amp.autocast():
                    loss = model.forward(input_ids = input_ids, labels = labels).loss
                
                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumilation_steps == 0:
                    scaler.step(optimizer)

                    scaler.update()
                    pbar.set_description(f"LOSS: {loss.item():.3f}")

                step += 1

    model.save_pretrained(save_directory = "./hf-models/usvsnsp-sft-model/")
    tokenizer.save_pretrained(save_directory = "./hf-models/usvsnsp-sft-model/")
    tokenizer.save_pretrained(save_directory = "./hf-models/usvsnsp-sft-combined/")

def deparallelize_save(parallel_context):
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neox-20b', cache_dir = './hf-models')
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
    train_model(parallel_context)
    torch.cuda.empty_cache()
    deparallelize_save(parallel_context)

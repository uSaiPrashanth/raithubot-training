import os
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import TensorParallel
from transformers import AutoModelForCausalLM, AutoTokenizer
import oslo

import torch
def generate(text):
    tokens = tokenizer([text], return_tensors = 'pt').input_ids.cuda()
    text = model.generate(
        input_ids = tokens, 
        do_sample = True,
        temperature = 0.5,
        top_k = 50,
        max_length = 1024,
        repetition_penalty = 1.1,
        use_cache = False
    )
    print(tokenizer.decode(text[0]))

if __name__ == '__main__':
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=int(os.environ['WORLD_SIZE']),
        tensor_parallel_mode=ParallelMode.TENSOR_1D,
    )
    model = AutoModelForCausalLM.from_pretrained("hf-models/usvsnsp-sft-combined", cache_dir = 'hf-models')
    tokenizer = AutoTokenizer.from_pretrained("hf-models/usvsnsp-sft-model", cache_dir = 'hf-models')

    model = TensorParallel(model, parallel_context)
    oslo.ready(model, parallel_context)

    model.from_parallelized(path = "./hf-models/usvsnsp-sft-model")

    generate(
        "Query: Hey I am a farmer but I am not able to understand what to do to stop my mango leaves from falling.\n"
        "Snippets: \n"
        "\t1.Plant was not growing and it leaves are becoming curly. ... Know all about proper fertilization to prevent deficiencies and improve your yield!\n"
        "\t2. 19-Jul-2022 — Solution – Keeping Mango trees well fertilized is the key to avoiding nutrient and mineral deficiency and avoiding your Mango tree dying slowly.\n"
        "\t3. 1. Collect the fallen fruits and destroy them. 2. Also harvest fruits early to reduce flies damage . 3. Use traps to monitor fruit flies. Traps can ...\n"
        "\t4. Potassium · Application of 1 kg muriate of potash or sulphate of potash along with 2 kg urea and 6 kg super phosphate during July-August in the basin could ...\n"
        "\t5. 03-May-2020 — Keep the area around the mango plant as weed-free as possible. And use a fungicide if the fungal problem persists. If the mango tree is already ...\n"
        "Response: "
    )
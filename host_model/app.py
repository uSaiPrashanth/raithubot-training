from flask import request, Flask
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

model = None
tokenizer = None
def generate(text):
    tokens = tokenizer([text], return_tensors = 'pt').input_ids.cuda()
    text = model.generate(
        input_ids = tokens, 
        do_sample = True,
        temperature = 0.2,
        top_k = 30,
        top_p = 0.8,
        max_length = 512,
        repetition_penalty = 1.1,
        use_cache = False
    )
    text = tokenizer.decode(text[0])
    for i in range(2):
        resp_start = text.find("Response:")
        text = text[resp_start + len("Response:"):]
    resp_end = text.find("Query:")
    if resp_end != -1:
        text = text[:resp_end]
    return text

@app.route('/raithuchat', methods=['GET', 'POST'])
def chat():
    prompt = request.args['prompt']
    response = generate(
        ("Query: I heard that the use of gram herbicide can cause negative effects on the environment. Can you explain more about this?\n"
        "Response: Gram herbicide is not a specific product name, but a term used to describe herbicides that are applied to crops that have been genetically modified to tolerate them, such as glyphosate, dicamba or 2,4-D. These herbicides can have negative effects on the environment by killing non-target plants and animals, contaminating water sources, reducing biodiversity and increasing the risk of herbicide resistance   . Some of these effects can be mitigated by following proper application guidelines, using alternative weed control methods and choosing environmentally friendly herbicide formulations .\n"
        "Query: {}\n"
        "Response:").format(prompt)
    )
    return {
        'response': response.strip()
    }

@app.route("/")
def help():
    return "Hello World"

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/pythia-2.8b',
        cache_dir = './hf-models',
    ).eval()
    model = PeftModel.from_pretrained(model, "hf-models/usvsnsp-sft-model").cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'hf-models/usvsnsp-sft-model', 
        cache_dir = './hf-models',
    )
    app.run(host = '10.180.0.2', port = 3232)

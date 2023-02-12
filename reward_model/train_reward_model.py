import yaml
import argparse
from transformers import DebertaV2Model, DebertaV2TokenizerFast

def parse_args():
    """Parses input arguments and returns config for training a reward model
    
    Loads the yaml file specified in input arguments.
    Required arguments for reward model are:
        reward_model: A pre-trained model from huggingface hub to be finetuned upon
    
    Returns:
        config: model config
    """
    parser = argparse.ArgumentParser(prog = 'Train Reward Model', description="Trains a reward model for RLHF")
    parser.add_argument("config")
    config_path = parser.parse_args().config
    
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config



if __name__ == '__main__':
    config = parse_args()
    model = DebertaV2Model.from_pretrained(config['reward_model'], cache_dir = config['cache_dir'])
    model = model.half().eval().cuda()
    tokenizer = DebertaV2TokenizerFast.from_pretrained(config['reward_model'], cache_dir = config['cache_dir'])

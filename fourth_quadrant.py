from utils.wrapper import *
import argparse
from utils.fourth_quadrant_processing import *
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model_name", type=str,default="Llama-2-7b-hf", help="model name")
    parser.add_argument("--dataset_path", type=str,default="dataset/emo", help="dataset name")
    parser.add_argument("--num_samples", type=int, default=50, help="number of samples to select")
    args = parser.parse_args()

    return args


def main(args):
    # Wrapper for different models
    wrapper=model_choose(args.model_name)

    #import dataset
    dataset = load_from_disk(args.dataset_path)
    train_data = dataset['train']
    train_dataset_list = list(train_data)

    # Calculate the accuracy
    if  "trec" in args. dataset_path:
        accuracy= fourth_quadrant_trec(wrapper,train_dataset_list,args)
        print(f"The proportion of incorrect label predictions by the {args.model_name} is: {accuracy}")
    
    elif "emo" in args.dataset_path:
        accuracy= fourth_quadrant_emo(wrapper,train_dataset_list,args)
        print(f"The proportion of incorrect label predictions by the {args.model_name} is: {accuracy}")
        
if __name__ == "__main__":
    args=parse_args()
    main(args)

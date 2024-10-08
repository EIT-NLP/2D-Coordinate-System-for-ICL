from utils.wrapper import *
import argparse
from utils.third_quadrant_processing import *
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model_name", type=str,default="Llama-2-7b-hf", help="model name")
    parser.add_argument("--dataset_path", type=str,default="dataset/trec", help="dataset name")
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

    # Calculate the prediction proportions for each position
    if  "trec" in args. dataset_path:
        proportions= third_quadrant_trec(wrapper,train_dataset_list,args)
        for i in range(6):
            print(f"Model: {args.model_name}, Proportion for position {i}: {proportions[i]}")
    
    elif "emo" in args.dataset_path:
        proportions= third_quadrant_emo(wrapper,train_dataset_list,args)
        for i in range(4):
            print(f"Model: {args.model_name}, Proportion for position {i}: {proportions[i]}")

if __name__ == "__main__":
    args=parse_args()
    main(args)

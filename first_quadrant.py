from utils.wrapper import *
import argparse
from utils.first_quadrant_processing import *
from datasets import load_from_disk
from dataset.Reasoning_about_Colored_Objects import *
from dataset.World_Capitals import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model_name", type=str,default="Llama-2-7b-hf", help="model name")
    parser.add_argument("--dataset_path", type=str,default="dataset/sst2", help="dataset name")
    parser.add_argument("--num_samples", type=int, default=50, help="number of samples to select")
    args = parser.parse_args()

    return args


def main(args):
    # Wrapper for different models
    wrapper=model_choose(args.model_name)

    # Calculate the accuracy
    if  "sst2" in args. dataset_path:
        #import dataset
        dataset = load_from_disk(args.dataset_path)
        train_data = dataset['train']
        train_dataset_list = list(train_data)
        
        # Calculate the accuracy for the SST-2 dataset.
        accuracy= first_quadrant_sst2 (wrapper, train_dataset_list, args)
        print(f"The {args.model_name}'s accuracy under the setting of Similar(T) with an incorrect label: ", accuracy)
    
    else:
        if "Color" in args.dataset_path:
            # Calculate the accuracy for the Reasoning about Colored Objects dataset.
            accuracy= first_quadrant_color_and_capital (wrapper, objects, colors)
        elif "Capital" in args.dataset_path:
            # Calculate the accuracy for the World Capitals dataset.
            accuracy= first_quadrant_color_and_capital (wrapper, countries, capitals)
        print(f"The {args.model_name}'s accuracy under the setting of Similar(T) with an incorrect label: ", accuracy)

if __name__ == "__main__":
    args=parse_args()
    main(args)

from utils.wrapper import *
import torch
from utils.plot import plot_PIR
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPT2LMHeadModel

def parse_args():
    parser = argparse.ArgumentParser(description="2D-Coordinate-System-for-ICL")
    
    # add arguments
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-hf", help="Name of the model to load") 
    parser.add_argument("--label_index", type=int, default=-1, help="Index of the label token")
    parser.add_argument("--plot_file", type=str, default="PIR.pdf", help="Name of the plot file")

    args = parser.parse_args()
    return args

def tokenize(text, wrapper):
        inp_ids = wrapper.tokenize(text)
        str_toks = wrapper.list_decode(inp_ids[0])
        return inp_ids, str_toks

def main(args):
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                torch_dtype=torch.float16, 
                                                attn_implementation="eager").to(device)

    # Create a wrapper for the model; one can follow the same pattern to create a wrapper for any model.
    if isinstance(model, LlamaForCausalLM):
        num_layers=model.config.num_hidden_layers
        wrapper = Llamawrapper(model, tokenizer,num_layers)
    elif isinstance(model, GPT2LMHeadModel):
        num_layers=model.config.n_layer
        wrapper = GPT2Wrapper(model, tokenizer,num_layers)
         
    # Example input text for the world capital task, which can be replaced with any other task.
    input_text="""Word: France
    Label: Paris
    Word: England
    Label: London
    Word: Germany
    Label: Berlin
    Word: Japan
    Label:"""

    # Tokenize the input text
    input_ids, input_toks = tokenize(input_text, wrapper)

    # Find the indices of the label tokens
    label_pre_indices = [i for i, token in enumerate(input_toks) if (token == ":" and (input_toks[i - 1] == "Label"))]
    label_pre_indices = label_pre_indices[:-1]  
    label_indices = [index+1 for index in label_pre_indices]

    # Choose the desired label token index
    token_index=label_indices[args.label_index]

    # Check if the desired label token is found
    print(input_toks[token_index])

    # Obtain the hidden states of all layers
    logits = wrapper.get_layers_logits(input_ids,token_index)

    # Calculate the reciprocal of the rank of the task-representative token within the vocabulary distribution for each layer
    rrs=wrapper.rr_per_layer(logits,"capital")

    # Plot the PIR 
    plot_PIR(rrs, args.plot_file)

if __name__ == "__main__":
    args=parse_args()
    main(args)

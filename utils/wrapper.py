from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

def model_choose(model_name):
    # The forthcoming updated version will incorporate support for both GPT-2 and GPT-J-6B models.
    if  "Llama" or "Mistral" in model_name: 
        model, tokenizer = load_model(model_name)
        num_layers=model.config.num_hidden_layers
        wrapper = Llamawrapper(model, tokenizer,num_layers)
    
    if  "gpt2" in model_name:
        model, tokenizer = load_model(model_name)
        num_layers=model.config.n_layer
        wrapper = GPT2Wrapper(model, tokenizer,num_layers)

    if  model_name =="gpt-j-6b" :   
        model, tokenizer = load_model(model_name)
        num_layers=model.config.n_layer
        wrapper = GPTJWrapper(model, tokenizer,num_layers)

    return wrapper

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def load_model(model_name):
    device= get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,attn_implementation="eager").to(device)
    return model, tokenizer


class ModelWrapper(nn.Module):

    def __init__(self, model, tokenizer,num_layers):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = get_device()
        self.num_layers = num_layers

    def tokenize(self, s):
        return self.tokenizer.encode(s, return_tensors='pt').to(self.device)

    def list_decode(self, inpids):
        return [self.tokenizer.decode(s) for s in inpids]
    
    def get_logit(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states=outputs.hidden_states
        hidden_states_final_layer=hidden_states[-1][0, -1, :]
        logit =torch.matmul(self.model.lm_head.weight.detach(), hidden_states_final_layer)
        return logit
    

class Llamawrapper(ModelWrapper):
    
    def layer_decode_hidden_state(self,hidden_states, attention_weights,token_index):
        # Create an empty list to store the hidden states of the label tokens for each layer.
        logits = []

        for layer_index in range(self.num_layers):     
            # Get the block, attention, and normalization layers of the model.
            block = self.model.model.layers[layer_index]
            attn = block.self_attn
            input_layernorm = block.input_layernorm
            ln_f = self.model.model.norm 
            
            # Obtain the hidden states of the layer.
            hidden_states_layer = hidden_states[layer_index]

            # Get the value vectors.
            normed = input_layernorm (hidden_states_layer)
            value = attn.v_proj(normed)
            bsz, q_len, _ = normed.size()
            value = value.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            # Get the attention weights of the layer.
            attn_weights = attention_weights[layer_index]

            # Get the attention weights of the label token.
            attn_weights_index = attn_weights[:, :, -1, token_index].unsqueeze(-1)

            # Get the value vectors of the label token.
            value_index = value[:, :, token_index, :]

            # Calculate the hidden state of the label token.
            hidden_state = attn_weights_index * value_index
            hidden_state = hidden_state.view(attn.num_heads * attn.head_dim)
            hidden_state = attn.o_proj(hidden_state.unsqueeze(0))
            hidden_state = ln_f(hidden_state)

            # Calculate the logits of the label token.
            logits_hidden_state = torch.matmul(self.model.lm_head.weight.detach(), hidden_state.T).squeeze(-1)

            # Append the logits of the label token to the list.
            logits.append(logits_hidden_state)

        return logits
    
    def get_layers_logits(self, input_ids, token_index,**kwargs):
        # get the hidden states and attention weights of all layers
        outputs = self.model(input_ids=input_ids,  
                            output_hidden_states=True,
                            output_attentions=True,
                            **kwargs)
        hidden_states=outputs.hidden_states
        attn_weights = outputs.attentions

        # get the logits of the label token for each layer
        logits = self.layer_decode_hidden_state(hidden_states,attn_weights,token_index)
        
        return torch.stack(logits)
    
    def rr_per_layer(self, logits, task_token):
        # tokenize the task token
        task_token_id = self.tokenizer.encode(task_token)[1]

        # create an empty list to store the reciprocal ranks of the task token for each layer
        rrs = []

        for i, layer in enumerate(logits):
            # calculate the softmax probabilities
            probs = F.softmax(layer,dim=-1)

            # sort the probabilities in descending order
            sorted_probs = probs.argsort(descending=True)

            # calculate the rank of the task token
            rank = float(np.where(sorted_probs.cpu().numpy()==task_token_id)[0][0])

            # Append the reciprocal rank of the task token to the list
            rrs.append(1/(rank+1))

        return np.array(rrs)


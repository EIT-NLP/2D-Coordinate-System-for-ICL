import torch.nn.functional as F
import torch
import random

def icl_generation_sst2(label_to_samples, test_sample, args):
    # Generate demonstration samples
    icl_template = ""
    count_shot_number = 0
    while count_shot_number < args.shot_number:
        label= random.choice(list(label_to_samples.keys()))
        sample = label_to_samples[label]
        sentence = random.choice(sample)['sentence']  
        icl_template += f"Sentence: {sentence}\nLabel: {label}\n"
        count_shot_number += 1

    # test_sample
    icl_template += f"Sentence: {test_sample}\nLabel:"
    
    return icl_template

def icl_generation_color_and_capital(data, dissimilar_labels_indices, labels, random_data):
    # Generate demonstration samples
    icl_template = ""
    for index in dissimilar_labels_indices:
        sentence = data[index]
        label = labels[index]
        icl_template += f"Sentence: {sentence}\nLabel: {label}\n"

    # test_sample
    icl_template += f"Sentence: {random_data}\nLabel:"
    
    return icl_template


def second_quadrant_sst2(wrapper, train_dataset_list, args):
    # Set the seed value, one can choose any seed value
    seeds=[44,22,35]

    # Initialize the results list
    results = []

    # Loop through the seed values
    for seed_value in seeds:
        # Set the seed value
        random.seed(seed_value)  
        torch.manual_seed(seed_value)  
        torch.cuda.manual_seed_all(seed_value)  

        # select random samples
        random_samples = random.sample(train_dataset_list, args.num_samples)

        # Remove the selected samples from the dataset
        remaining_data = [sample for sample in train_dataset_list if sample not in random_samples]

        # Create samples based on label after removing the selected samples
        negative_samples = [sample for sample in remaining_data if sample['label'] == 0]
        positive_samples = [sample for sample in remaining_data if sample['label'] == 1]

        # Define a mapping between labels and their corresponding sample sets
        label_to_samples = {
            "negative": negative_samples,
            "positive": positive_samples,
        }

        # Perform ICL for selected samples
        for index in range(len(random_samples)):
            
            # Select a random sample as test sample
            random_data = random_samples[index]
            random_data_text=random_data['sentence']
            random_data_label = random_data['label']
            correct_label = 'negative' if random_data_label == 0 else 'positive'

            # Generate ICL
            icl_sst2 = icl_generation_sst2(label_to_samples, random_data_text,args)

            # Perform ICL
            result = run_icl_second_quadrant(icl_sst2, correct_label, wrapper) 

            # Append the result to the results list
            results.append(result)

    # Calculate the accuracy
    total = len(results)
    accuracy = sum(results) / total
    
    return accuracy

def second_quadrant_color_and_capital(wrapper, data, labels, args):
    # Set the seed value, one can choose any seed value
    seeds=[44,22,35]

    # Initialize the results list
    results = []

    # Loop through the seed values
    for seed_value in seeds:
        # Set the seed value
        random.seed(seed_value)  
        torch.manual_seed(seed_value)  
        torch.cuda.manual_seed_all(seed_value)  

        # Perform ICL for selected samples
        for index in range(len(data)):
            
            # Select a random sample as test sample
            random_data = data[index]
            correct_label = labels[index]

            # Select {args.shot_number} dissimilar examples
            dissimilar_labels_indices= random.sample([i for i in range(len(data)) if i != index], args.shot_number)
            
            # Generate ICL
            icl = icl_generation_color_and_capital(data, dissimilar_labels_indices, labels, random_data)

            # Perform ICL
            result = run_icl_second_quadrant(icl, correct_label, wrapper) 

            # Append the result to the results list
            results.append(result)

    # Calculate the accuracy
    total = len(results)
    accuracy = sum(results) / total
    
    return accuracy

def tokenize(wrapper, text):
        inp_ids = wrapper.tokenize(text)
        str_toks = wrapper.list_decode(inp_ids[0])
        return inp_ids, str_toks

def run_icl_second_quadrant(text, label, wrapper):
    # Tokenize the input text
    input_ids, _ = tokenize(wrapper, text)

    # Get the logits
    logit = wrapper.get_logit(input_ids)
    
    # Get the prediction
    prediction= wrapper.tokenizer.decode(F.softmax(logit,dim=-1).argsort(descending=True)[:1]) 

    # Check if the model predicts the label
    return 1 if label.startswith(prediction.lstrip()) else 0
    
import torch.nn.functional as F
import torch
import random
import re

def icl_generation(label_to_samples,test_sample):
    # Generate demonstration samples
    icl_template = ""
    for i in range(2):
        for label, sample in label_to_samples.items():
            sentence = random.choice(sample)['text']  
            icl_template += f"Sentence: {sentence}\nLabel: {label}\n"

    # test_sample
    icl_template += f"Sentence: {test_sample}\nLabel:"
    
    return icl_template

def third_quadrant_emo(wrapper, train_dataset_list, args):
    # initialize the counts
    counts = [0] * 4

    # Set the seed value, one can choose any seed value
    seeds=[44,22,35]

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
        Others_samples = [sample for sample in remaining_data if sample['label'] == 0]
        Happy_samples = [sample for sample in remaining_data if sample['label'] == 1]
        Sad_samples = [sample for sample in remaining_data if sample['label'] == 2]
        Angry_samples = [sample for sample in remaining_data if sample['label'] == 3]

        # Define a mapping between labels and their corresponding sample sets
        label_to_samples = {
            "others": Others_samples,
            "happy": Happy_samples,
            "sad": Sad_samples,
            "angry": Angry_samples,
        }

        # Perform ICL for selected samples
        for index in range(len(random_samples)):
            
            # Select a random sample as test sample
            random_data = random_samples[index]
            random_data_text=random_data['text']
            
            # Generate ICL
            icl_emo = icl_generation(label_to_samples, random_data_text)

            # Perform ICL
            results = run_icl_third_quadrant(icl_emo, wrapper, args) 

            # count for each position
            for i, result in enumerate(results):
                counts[i] += result 

    # Calculate the proportions 
    total = sum(counts)
    propotions = [count / total for count in counts]
    
    return propotions

def third_quadrant_trec(wrapper, train_dataset_list, args):
    # initialize the counts
    counts = [0] * 6

    # Set the seed value, one can choose any seed value
    seeds=[44,22,35]

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

        # Create samples based on coarse_label after removing the selected samples
        Abbreviation_samples = [sample for sample in remaining_data if sample['coarse_label'] == 0]
        Entity_samples = [sample for sample in remaining_data if sample['coarse_label'] == 1]
        Description_samples = [sample for sample in remaining_data if sample['coarse_label'] == 2]
        Human_samples = [sample for sample in remaining_data if sample['coarse_label'] == 3]
        Location_samples = [sample for sample in remaining_data if sample['coarse_label'] == 4]
        Numeric_samples = [sample for sample in remaining_data if sample['coarse_label'] == 5]

        # Define a mapping between labels and their corresponding sample sets
        label_to_samples = {
            "Abbreviation": Abbreviation_samples,
            "Entity": Entity_samples,
            "Description": Description_samples,
            "Human": Human_samples,
            "Location": Location_samples,
            "Number": Numeric_samples
        }
                                                 
        # Perform ICL for selected samples
        for index in range(len(random_samples)):
            
            # Select a random sample as test sample
            random_data = random_samples[index]
            random_data_text=random_data['text']

            # Generate ICL
            icl_trec = icl_generation(label_to_samples, random_data_text)

            # Perform ICL
            results = run_icl_third_quadrant(icl_trec, wrapper, args) 

            # count for each position
            for i, result in enumerate(results):
                counts[i] += result 

    # Calculate the proportions 
    total = sum(counts)
    propotions = [count / total for count in counts]
    
    return propotions

def tokenize(wrapper, text):
        inp_ids = wrapper.tokenize(text)
        str_toks = wrapper.list_decode(inp_ids[0])
        return inp_ids, str_toks

def run_icl_third_quadrant(text,wrapper,args):
    # Tokenize the input text
    input_ids, _ = tokenize(wrapper, text)

    # Get the logits
    logit = wrapper.get_logit(input_ids)
    
    # Get the prediction
    prediction= wrapper.tokenizer.decode(F.softmax(logit,dim=-1).argsort(descending=True)[:1]) 

    # Extract the labels
    labels = re.findall(r'Label: (\w+)', text)

    if "trec" in args.dataset_path:
        # Initialize the results
        results=[0] * 6

        # Check label of which position the model predicts
        for i in range(6):
            if labels[i].startswith(prediction.lstrip()):
                results[i] += 1

        return results
    
    elif "emo" in args.dataset_path:
        # Initialize the results
        results=[0] * 4

        # Check label of which position the model predicts
        for i in range(4):
            if labels[i].startswith(prediction.lstrip()):
                results[i] += 1

        return results
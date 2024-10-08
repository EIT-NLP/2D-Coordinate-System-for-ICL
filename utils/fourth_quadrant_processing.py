import torch.nn.functional as F
import torch
import random

def icl_generation(label_to_samples, test_sample):
    # Generate demonstration samples
    icl_template = ""
    for i in range(2):
        for label, sample in label_to_samples.items():
            sentence = random.choice(sample)['text']  
            icl_template += f"Sentence: {sentence}\nLabel: {label}\n"

    # test_sample
    icl_template += f"Sentence: {test_sample}\nLabel:"
    
    return icl_template

def fourth_quadrant_emo(wrapper, train_dataset_list, args):
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
            correct_label = random_data['label']

            # List of all possible labels except for the test sample's label
            label_list = ['others', 'happy', 'sad', 'angry']
            label_list.pop(correct_label)  # Remove the test sample's label

            # Randomly select a label different from correct_label
            incorrect_label = random.choice(label_list)

            # Generat ICL
            icl_emo = icl_generation(label_to_samples, random_data_text)

            # Split the ICL into sentences list
            sentences = icl_emo.split("\n")

            # Replace the sentence with the label of incorrect_label
            for i in range(len(sentences)):
                if f"Label: {incorrect_label}" in sentences[i] and i > 0:
                    sentences[i-1] = f"Sentence: {random_data_text}"  # Replace sentence with random_data_text
            
            # Merge the sentences back into a single text
            text = "\n".join(sentences)

            # Perform ICL
            result = run_icl_fourth_quadrant(text, incorrect_label, wrapper) 

            # Append the result to the results list
            results.append(result)

    # Calculate the accuracy
    total = len(results)
    accuracy = sum(results) / total
    
    return accuracy

def fourth_quadrant_trec(wrapper, train_dataset_list, args): 
    # Set the seed value, one can choose any seed value
    seeds=[44,22,35]

    # initialize the results list
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
            correct_label = random_data['coarse_label']

            # List of all possible labels except for the test sample's label
            label_list = ['Abbreviation', 'Entity', 'Description', 'Human', 'Location', 'Number']
            label_list.pop(correct_label)  # Remove the test sample's label

            # Randomly select a label different from correct_label
            incorrect_label = random.choice(label_list)

            # Generat ICL
            icl_trec = icl_generation(label_to_samples, random_data_text)

            # Split the ICL into sentences list
            sentences = icl_trec.split("\n")
            
            # Replace the sentence with the label of incorrect_label
            for i in range(len(sentences)):
                if f"Label: {incorrect_label}" in sentences[i] and i > 0:
                    sentences[i-1] = f"Sentence: {random_data_text}"  # Replace sentence with random_data_text

            # Merge the sentences back into a single text
            text = "\n".join(sentences)

            # Perform ICL
            result = run_icl_fourth_quadrant(text, incorrect_label, wrapper) 

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

def run_icl_fourth_quadrant(text, label, wrapper):
    # Tokenize the input text
    input_ids, _ = tokenize(wrapper, text)

    # Get the logits
    logit = wrapper.get_logit(input_ids)
    
    # Get the prediction
    prediction= wrapper.tokenizer.decode(F.softmax(logit,dim=-1).argsort(descending=True)[:1]) 

    # Check if the model predicts the label
    return 1 if label.startswith(prediction.lstrip()) else 0
    
import torch.nn.functional as F
import torch
import random

def icl_generation_sst2(label_to_samples, test_sample):
    # Generate demonstration samples
    icl_template = ""
    for i in range(2):
        for label, sample in label_to_samples.items():
            sentence = random.choice(sample)['sentence']  
            icl_template += f"Sentence: {sentence}\nLabel: {label}\n"

    # test_sample
    icl_template += f"Sentence: {test_sample}\nLabel:"
    
    return icl_template

def template_generation_color_and_capital(data, labels):
    # Generate ICL
    icl_template = ""
    for index in random.sample(range(len(data)), 2):
        sentence = data[index]
        label = labels[index]
        icl_template += f"Sentence: {sentence}\nLabel: {label}\n"

    # test_sample
    test_sample_index = random.choice(range(len(data)))
    test_sentence = data[test_sample_index]
    icl_template += f"Sentence: {test_sentence}\nLabel:"
    
    return icl_template


def first_quadrant_sst2(wrapper, train_dataset_list, args):
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

            # choose the label different from the test sample's label as incorrect_label
            correct_label = 'negative' if random_data_label == 0 else 'positive'
            incorrect_label = 'positive' if random_data_label == 0 else 'negative'
            
            # Generate ICL
            icl_sst2 = icl_generation_sst2(label_to_samples, random_data_text)
            
            # Split the ICL into sentences list
            sentences = icl_sst2.split("\n")

            # Replace the sentence with the label of incorrect_label
            for i in range(len(sentences)):
                if f"Label: {incorrect_label}" in sentences[i] and i > 0:
                    sentences[i-1] = f"Sentence: {random_data_text}"  # Replace sentence with random_data_text
            
            # Merge the sentences back into a single text
            text = "\n".join(sentences)

            # Perform ICL
            result = run_icl_first_quadrant(text, correct_label, wrapper) 

            # Append the result to the results list
            results.append(result)

    # Calculate the accuracy
    total = len(results)
    accuracy = sum(results) / total
    
    return accuracy

def first_quadrant_color_and_capital(wrapper, data, labels):
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

        # Generate template for ICL
        template = template_generation_color_and_capital(data, labels)

        # Perform ICL for selected samples
        for index in range(len(data)):
            
            # Select a random sample as test sample
            random_data = data[index]
            correct_label = labels[index]

            # choose the label different from the test sample's label as incorrect_label
            incorrect_labels_indices= random.sample([i for i in range(len(data)) if i != index], 2)
            incorrect_label_index= incorrect_labels_indices[0]
            incorrect_label = labels[incorrect_label_index]

            # Split the input text into sentences list
            sentences = template.split("\n")
            
            # Construct ICL
            count_incorrect_label = 0
            for i in range(len(sentences)-1):
                if "Label:" in sentences[i]:
                    sentences[i-1] = f"Sentence: {data[incorrect_labels_indices[count_incorrect_label%2]]}"  
                    sentences[i] = f"Label: {labels[incorrect_labels_indices[count_incorrect_label%2]]}"
                    count_incorrect_label += 1

            # test sample
            if sentences[-2].startswith("Sentence:"):
                sentences[-2] = f"Sentence: {random_data}"

            # Replace the sentence with the label of incorrect_label
            for i in range(len(sentences)):
                if f"Label: {incorrect_label}" in sentences[i] and i > 0:
                    sentences[i-1] = f"Sentence: {random_data}"  # Replace sentence with random_data_text
            
            # Merge the sentences back into a single text
            text = "\n".join(sentences)

            # Perform ICL
            result = run_icl_first_quadrant(text, correct_label, wrapper) 

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

def run_icl_first_quadrant(text, label, wrapper):
    # Tokenize the input text
    input_ids, _ = tokenize(wrapper, text)

    # Get the logits
    logit = wrapper.get_logit(input_ids)
    
    # Get the prediction
    prediction= wrapper.tokenizer.decode(F.softmax(logit,dim=-1).argsort(descending=True)[:1]) 

    # Check if the model predicts the label
    return 1 if label.startswith(prediction.lstrip()) else 0
    
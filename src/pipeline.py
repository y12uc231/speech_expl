import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from datasets import load_dataset

# Load the Common Voice dataset
dataset = load_dataset("common_voice", "en", split="train[:100]")

# Load the Wav2Vec2 model and processor
model_id = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2Model.from_pretrained(model_id)

def compute_shapley_interaction(features, t1, t2, t3):
    set_A = features[t1].unsqueeze(0)
    set_B = features[t2].unsqueeze(0)
    empty_set = torch.zeros_like(set_A)

    # Compute STII using Equation 3
    stii = torch.norm(model(empty_set).last_hidden_state - model(set_A).last_hidden_state - 
                      model(set_B).last_hidden_state + model(torch.cat([set_A, set_B])).last_hidden_state) / torch.norm(model(empty_set).last_hidden_state)
    return stii.item()

def compute_average_interaction(features, t_boundary, delta):
    # Compute average interaction using Equation 6
    interaction_sum = 0
    count = 0
    for t1 in range(max(0, t_boundary - delta), min(len(features), t_boundary + delta)):
        t2 = t1 + 1
        if t2 < len(features):
            t3 = t2 + 1
            interaction_sum += compute_shapley_interaction(features, t1, t2, t3)
            count += 1
    return interaction_sum / count if count > 0 else 0

# Experiment 1: Vowels and Consonants
vowel_consonant_interactions = []
consonant_consonant_interactions = []

for example in dataset:
    # Preprocess the audio and extract features
    audio = example["audio"]["array"]
    features = processor(audio, sampling_rate=example["audio"]["sampling_rate"], return_tensors="pt").input_values[0]
    
    # Align the audio with phonemes
    phonemes = processor.tokenizer.sp.encode(example["sentence"], out_type=str)
    phoneme_boundaries = processor.tokenizer.sp.encode(example["sentence"], out_type=str, enable_sampling=True)[1].numpy().tolist()
    
    for i in range(len(phoneme_boundaries) - 1):
        t_boundary = phoneme_boundaries[i]
        delta = 0.1 * (phoneme_boundaries[i+1] - phoneme_boundaries[i])
        
        if phonemes[i] in ['a', 'e', 'i', 'o', 'u'] and phonemes[i+1] not in ['a', 'e', 'i', 'o', 'u']:
            vowel_consonant_interactions.append(compute_average_interaction(features, t_boundary, delta))
        elif phonemes[i] not in ['a', 'e', 'i', 'o', 'u'] and phonemes[i+1] not in ['a', 'e', 'i', 'o', 'u']:
            consonant_consonant_interactions.append(compute_average_interaction(features, t_boundary, delta))

# Compute average interactions for vowel-consonant and consonant-consonant transitions
avg_vowel_consonant_interaction = sum(vowel_consonant_interactions) / len(vowel_consonant_interactions)
avg_consonant_consonant_interaction = sum(consonant_consonant_interactions) / len(consonant_consonant_interactions)

print(f"Average vowel-consonant interaction: {avg_vowel_consonant_interaction}")
print(f"Average consonant-consonant interaction: {avg_consonant_consonant_interaction}")

# Experiment 2: Manner of Consonant Articulation
consonant_interactions = {}

for example in dataset:
    # Preprocess the audio and extract features
    audio = example["audio"]["array"]
    features = processor(audio, sampling_rate=example["audio"]["sampling_rate"], return_tensors="pt").input_values[0]
    
    # Align the audio with phonemes
    phonemes = processor.tokenizer.sp.encode(example["sentence"], out_type=str)
    phoneme_boundaries = processor.tokenizer.sp.encode(example["sentence"], out_type=str, enable_sampling=True)[1].numpy().tolist()
    
    for i in range(len(phoneme_boundaries) - 1):
        t_boundary = phoneme_boundaries[i]
        delta = 0.1 * (phoneme_boundaries[i+1] - phoneme_boundaries[i])
        
        if phonemes[i] not in ['a', 'e', 'i', 'o', 'u']:
            if phonemes[i] not in consonant_interactions:
                consonant_interactions[phonemes[i]] = []
            consonant_interactions[phonemes[i]].append(compute_average_interaction(features, t_boundary, delta))

# Compute average interactions for each consonant
for consonant, interactions in consonant_interactions.items():
    avg_interaction = sum(interactions) / len(interactions)
    print(f"Average interaction for {consonant}: {avg_interaction}")

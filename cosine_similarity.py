import os
import torch
import csv
from torch.nn.functional import cosine_similarity

def load_language_reps(lang, representations_dir="representations_hubert"):
    """
    Loads the precomputed average representations for a given language from disk.
    Returns a tensor of shape (num_clips, num_layers, hidden_size).
    """
    file_path = os.path.join(representations_dir, f"{lang}_avg_reps.pt")
    reps = torch.load(file_path)
    return reps

def compute_layerwise_cosine_similarity(ref_reps, tgt_reps):
    """
    Computes the layer-wise cosine similarity between two sets of representations:
    - ref_reps: (num_clips, num_layers, hidden_size)
    - tgt_reps: (num_clips, num_layers, hidden_size)

    Assumes the two sets have the same number of clips (i.e., matched pairs).
    Returns a list [sim_layer_0, sim_layer_1, ..., sim_layer_(num_layers-1)].
    """
    num_clips, num_layers, hidden_size = ref_reps.shape
    # We'll store the similarity of each layer across all clips
    similarities_per_layer = [[] for _ in range(num_layers)]

    for i in range(num_clips):
        # shape: (num_layers, hidden_size)
        ref_clip = ref_reps[i]
        tgt_clip = tgt_reps[i]
        for layer in range(num_layers):
            # shape: (hidden_size,)
            ref_layer = ref_clip[layer]
            tgt_layer = tgt_clip[layer]
            # Compute cosine similarity
            sim = cosine_similarity(ref_layer.unsqueeze(0), tgt_layer.unsqueeze(0), dim=1)
            similarities_per_layer[layer].append(sim.item())

    # Average similarity per layer
    avg_similarities = [
        sum(similarities_per_layer[layer]) / len(similarities_per_layer[layer])
        for layer in range(num_layers)
    ]
    return avg_similarities

def main():
    # The language codes used previously
    language_codes = {
        "catalan": "ca_es",
        "spanish": "es_419",
        "french":  "fr_fr",
        "occitan": "oc_fr",
        "italian": "it_it",
        "german":  "de_de"
    }
    
    # Load the reference language representations (Catalan here)
    ref_lang = "catalan"
    ref_reps = load_language_reps(ref_lang)

    # We'll compute similarities with each of these
    target_langs = [lang for lang in language_codes if lang != ref_lang]

    # Dictionary to hold layer-wise similarities for each language
    layerwise_similarities = {}

    for lang in target_langs:
        print(f"Computing similarities for {lang} vs {ref_lang}...")
        tgt_reps = load_language_reps(lang)

        # Make sure both have the same number of clips to zip over
        # If they differ, you might need to handle that (e.g., clip to min length).
        min_clips = min(ref_reps.shape[0], tgt_reps.shape[0])
        ref_reps_sub = ref_reps[:min_clips]
        tgt_reps_sub = tgt_reps[:min_clips]

        # Compute layerwise similarity
        avg_sims = compute_layerwise_cosine_similarity(ref_reps_sub, tgt_reps_sub)
        layerwise_similarities[lang] = avg_sims

    # Write results to a CSV
    output_file = "cosine_similarities_Hubert_Fleurs.csv"
    num_layers = ref_reps.shape[1]
    header = ["Language"] + [f"Layer {i+1}" for i in range(num_layers)]

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for lang, sims in layerwise_similarities.items():
            writer.writerow([lang] + sims)
    
    print(f"Cosine similarities saved to {output_file}.")

if __name__ == "__main__":
    main()

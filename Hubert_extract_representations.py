import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertModel

def load_fleurs_test_split(language_code):
    """
    Load the test split of the Fleurs dataset for a specific language code.
    Returns a list of audio waveforms (each item is a dict with 'array', 'sampling_rate', etc.).
    """
    dataset = load_dataset("google/fleurs", language_code, split="test", trust_remote_code=True)
    return list(dataset["audio"])


def get_avg_hidden_representations(audio_list, feature_extractor, model, device="cuda"):
    """
    Given a list of audio waveforms, compute the *average* hidden representation
    for each layer. Returns a tensor of shape (num_audio_clips, num_layers, hidden_size).

    - Each item in audio_list is typically a dict with keys like 'array' and 'sampling_rate'.
    - For each clip, we average across the time dimension to get a single vector per layer.
    """
    all_avg_reps = []

    for audio in tqdm(audio_list, desc="Processing Audio Clips"):
        # If audio is a dict with 'array', extract the raw waveform
        if isinstance(audio, dict) and "array" in audio:
            audio_samples = audio["array"]
            sampling_rate = audio["sampling_rate"]
        else:
            # If it's just a NumPy array, you might need a default sampling rate
            audio_samples = audio
            sampling_rate = 16000  # or whatever your default is

        # Convert raw audio to model inputs
        inputs = feature_extractor(audio_samples, 
                                   sampling_rate=sampling_rate, 
                                   return_tensors="pt")
        # Move inputs to the correct device
        input_values = inputs["input_values"].to(device)  # shape: (batch_size=1, seq_len)

        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
            # outputs.hidden_states is a tuple of length [num_layers + 1],
            # because some models include the embedding layer as hidden_states[0].
            # For HuBERT base, you typically have 13 layers (12 transformer blocks + 1 embedding layer).
            hidden_states = torch.stack(outputs.hidden_states)  # (num_layers+1, batch_size=1, seq_len, hidden_size)

        # If you want to ignore the embedding layer, you could slice off hidden_states[1:].
        # For demonstration, let's keep everything. 
        # We'll also remove batch dimension (dim=1).
        hidden_states = hidden_states.squeeze(dim=1)  # shape: (num_layers+1, seq_len, hidden_size)

        # Average across the time (seq_len) dimension
        # shape -> (num_layers+1, hidden_size)
        avg_hidden_states = hidden_states.mean(dim=1)

        # Move to CPU to save memory
        avg_hidden_states = avg_hidden_states.cpu()

        # Collect for this clip
        all_avg_reps.append(avg_hidden_states)

    # Stack them into a single tensor of shape (num_clips, num_layers+1, hidden_size)
    all_avg_reps = torch.stack(all_avg_reps, dim=0)
    return all_avg_reps


def main():
    # You can change this to a multilingual or a larger HuBERT model if desired
    # e.g. "facebook/hubert-large-ls960-ft" or "facebook/hubert-base-ls960"
    model_name = "facebook/hubert-base-ls960"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(device)

    # Example: dictionary of languages (Fleurs codes).
    language_codes = {
        "catalan": "ca_es",
        "spanish": "es_419",
        "french":  "fr_fr",
        "occitan": "oc_fr",
        "italian": "it_it",
        "german":  "de_de"
    }

    output_dir = "representations_hubert"
    os.makedirs(output_dir, exist_ok=True)

    for lang, lang_code in language_codes.items():
        print(f"\n--- Processing {lang} ({lang_code}) ---")
        audio_list = load_fleurs_test_split(lang_code)
        
        # Optionally limit the number of clips for debugging
        # audio_list = audio_list[:50]

        avg_reps = get_avg_hidden_representations(audio_list, feature_extractor, model, device=device)

        # Save the representations to disk
        output_path = os.path.join(output_dir, f"{lang}_avg_reps.pt")
        torch.save(avg_reps, output_path)
        print(f"Saved representations for {lang} to {output_path}")


if __name__ == "__main__":
    main()

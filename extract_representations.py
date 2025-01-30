import os
import torch
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
from tqdm import tqdm


def load_fleurs_test_split(language_code):
    """
    Load the test split of the Fleurs dataset for a specific language code.
    Returns a list of audio waveforms (each item is a dict with 'array', 'sampling_rate', etc.).
    """
    dataset = load_dataset("google/fleurs", language_code, split="test", trust_remote_code=True)
    return list(dataset["audio"])


def get_avg_hidden_representations(audio_list, processor, model, device="cuda"):
    """
    Given a list of audio waveforms, compute the *average* hidden representation
    for each layer. Returns a list of Tensors of shape (num_audio_clips, num_layers, hidden_size).
    
    - Each element in the returned list corresponds to:
        hidden_states[i] -> (num_layers, hidden_size)
      for the i-th audio clip.
    """
    all_avg_reps = []

    for audio in tqdm(audio_list, desc="Processing Audio Clips"):
        # If audio is a dict with 'array', extract the raw waveform
        if isinstance(audio, dict) and "array" in audio:
            audio = audio["array"]

        # Preprocess with Whisper's processor
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to(device)

        # Pass through the model
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
        
        # outputs.hidden_states is a tuple of length num_layers
        # each entry is (batch_size=1, seq_len, hidden_size)
        # We stack them so shape is (num_layers, 1, seq_len, hidden_size)
        hidden_states = torch.stack(outputs.hidden_states)

        # Average across the sequence length (dim=2) and remove batch dimension
        # shape becomes: (num_layers, hidden_size)
        avg_hidden_states = hidden_states.squeeze(1).mean(dim=1)

        all_avg_reps.append(avg_hidden_states.cpu())  # Move to CPU to save memory

    # Stack them into a single tensor of shape (num_audio_clips, num_layers, hidden_size)
    # If you expect many audio clips, keep it as a list if that’s more convenient for you.
    all_avg_reps = torch.stack(all_avg_reps, dim=0)  # (num_clips, num_layers, hidden_size)

    return all_avg_reps


def main():
    # You can change this to other Whisper models if you want
    model_name = "openai/whisper-medium.en"

    # Initialize processor and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).encoder.to(device)

    # Example: dictionary of languages. Feel free to limit to one at a time.
    language_codes = {
        "catalan": "ca_es",
        "spanish": "es_419",
        "french":  "fr_fr",
        "occitan": "oc_fr",
        "italian": "it_it",
        "german":  "de_de"
    }

    # Directory to store the representations
    output_dir = "representations"
    os.makedirs(output_dir, exist_ok=True)

    # Process each language *separately*
    for lang, lang_code in language_codes.items():
        print(f"\n--- Processing {lang} ---")
        audio_list = load_fleurs_test_split(lang_code)
        
        # Optionally limit the number of clips to save time (e.g., debug)
        # audio_list = audio_list[:50]  # <-- uncomment to restrict for testing

        avg_reps = get_avg_hidden_representations(audio_list, processor, model, device=device)

        # Save the representations to disk
        output_path = os.path.join(output_dir, f"{lang}_avg_reps_medium.pt")
        torch.save(avg_reps, output_path)
        print(f"Saved representations for {lang} to {output_path}")


if __name__ == "__main__":
    main()

import torch
import json
import argparse  # Import argparse
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, Dataset  # DataLoader might not be strictly needed if processing one by one

# Assuming Model.py and Hype.py are in the same directory or accessible in PYTHONPATH
from Model import HateSpeechDetectionModel
from Hype import *
from eval import convert_logits_to_quads  # Tentative import


def main():
    # --- 0. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run hate speech detection and output results.")
    parser.add_argument('--input_file', type=str, default='test1.json',
                        help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, default='results.txt',
                        help='Path to save the output TXT file.')
    # Construct default model path using Hype.py constants
    default_model_path = f"{MODEL_SAVE_PATH}_epoch_{EPOCH}.pth"
    parser.add_argument('--model_file', type=str, default=default_model_path,
                        help=f'Path to the trained model file (.pth). Default: {default_model_path}')
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Tokenizer & Model ---
    print("Loading tokenizer and model...")
    tokenizer_name = 'bert-base-chinese'  # Or load from a config if varies
    model_bert_name = 'bert-base-chinese'  # Or load from a config if varies

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    model = HateSpeechDetectionModel(bert_model_name=model_bert_name)

    # Use model_file from argparse
    model_file_path = args.model_file

    try:
        print(f"Attempting to load model weights from: {model_file_path}")
        # Load state dict, ensuring it's mapped to the correct device
        state_dict = torch.load(model_file_path, map_location=device)

        # Adjust state_dict keys if necessary (e.g. if saved with DataParallel, keys might have 'module.' prefix)
        # Example: new_state_dict = {k.replace('module.',''): v for k, v in state_dict.items()}
        # For now, assume direct loading works.
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        print("Tokenizer and model loaded successfully.")
    except FileNotFoundError:
        print(
            f"Error: Model file not found at {model_file_path}. Please check the path and ensure the model is trained and saved.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # --- 2. Load Data ---
    # Use input_file from argparse
    print(f"Loading data from {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data_to_predict = json.load(f)  # Assumes a list of items
        # Ensure data_to_predict is a list, even if the file contains a single JSON object.
        if not isinstance(data_to_predict, list):
            data_to_predict = [data_to_predict]
        print(f"Data loaded successfully. Number of items: {len(data_to_predict)}")
    except FileNotFoundError:
        print(f"Error: Input data file not found at {args.input_file}.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_file}.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    if not data_to_predict:
        print("No data to predict. Exiting.")
        exit(0)

    # --- 3. Perform Prediction ---
    print("Performing predictions...")
    all_predictions_formatted = []  # Will store lines for the output file

    with torch.no_grad():  # Ensure no gradients are calculated during inference
        for item_idx, item in enumerate(data_to_predict):
            print(f"Processing item {item_idx + 1}/{len(data_to_predict)}...")
            # Assuming each item is a dict and has a 'content' field for the text.
            # Modify 'content' if your JSON structure uses a different key.
            text_content = item.get('content')
            if text_content is None:
                print(f"Warning: Item {item_idx + 1} does not have a 'content' field. Skipping.")
                # Add a SEP even for skipped items if required by output format, or handle as error.
                # For now, just skipping actual prediction for this item.
                # If an empty prediction is needed:
                # all_predictions_formatted.append("SEP")
                continue

            # Tokenize the input text
            # The tokenizer should handle padding and truncation to MAX_SEQ_LENGTH
            inputs = tokenizer(
                text_content,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',  # Pad to max_length
                truncation=True,  # Truncate if longer than max_length
                return_tensors="pt"  # Return PyTorch tensors
            )

            input_ids = inputs['input_ids'].to(device)  # Shape: [1, MAX_SEQ_LENGTH]
            attention_mask = inputs['attention_mask'].to(device)  # Shape: [1, MAX_SEQ_LENGTH]
            # BERT models generally expect token_type_ids as well
            token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids)).to(
                device)  # Shape: [1, MAX_SEQ_LENGTH]

            # Model forward pass
            model_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Prepare outputs for convert_logits_to_quads (expects single sample outputs)
            # The model outputs are already batched (batch_size=1 here)
            # Ensure all required keys by convert_logits_to_quads are present.
            # convert_logits_to_quads expects 'target_start_logits', 'target_end_logits', etc.
            # and 'sequence_output', 'cls_output', 'input_ids', 'attention_mask'.
            # The model output dict should directly provide these for the single sample.
            # We also need to pass the original input_ids and attention_mask for convert_logits_to_quads
            outputs_for_quad_conversion = {
                'target_start_logits': model_outputs['target_start_logits'],  # Already [1, L, 1] or [1,L] from model
                'target_end_logits': model_outputs['target_end_logits'],
                'argument_start_logits': model_outputs['argument_start_logits'],
                'argument_end_logits': model_outputs['argument_end_logits'],
                'sequence_output': model_outputs['sequence_output'],  # Already [1, L, H]
                'cls_output': model_outputs['cls_output'],  # Already [1, H]
                'input_ids': input_ids,  # Pass the tokenized input_ids [1, L]
                'attention_mask': attention_mask  # Pass the attention_mask [1, L]
            }

            # Convert logits to quads
            # MAX_SEQ_LENGTH is available from Hype.py import
            predicted_quads_for_sample = convert_logits_to_quads(
                outputs_for_quad_conversion,
                input_ids,  # sample_input_ids for decoding text from tokens
                tokenizer,
                MAX_SEQ_LENGTH,  # seq_len for span generation logic inside convert_logits_to_quads
                model  # model instance for _get_span_representation and classifiers
            )

            # Format quads for output
            if predicted_quads_for_sample:
                # Join multiple quads for the same sample with " [SEP] "
                sample_output_str = " [SEP] ".join(predicted_quads_for_sample)
            else:
                # If no quads are predicted for a sample
                sample_output_str = ""  # Start with an empty string

            # Add [END] marker to this specific sample's output string
            sample_output_str += " [END]"
            all_predictions_formatted.append(sample_output_str)

    # Global "SEP" and "END" markers are removed as per new formatting.
    print("Predictions complete.")

    # --- 4. Save Results ---
    # Use output_file from argparse
    print(f"Saving results to {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for line in all_predictions_formatted:
                f.write(line + "\n")
        print(f"Results saved successfully to {args.output_file}.")
    except Exception as e:
        print(f"Error saving results: {e}")
        exit(1)


if __name__ == '__main__':
    main()

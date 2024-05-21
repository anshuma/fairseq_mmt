import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/hf-seamless-m4t-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/hf-seamless-m4t-large")

# Ensure the model is in evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def translate(text, src_lang, tgt_lang):
    # Prepare the input text
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)

    # Generate translation
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True, tgt_lang=tgt_lang)

    # Decode the generated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text


def read_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)



src_lang = 'en'
tgt_lang = 'deu'

input_file = 'small_dataset/data/multi30k-en-de/train.en'  # Path to the input file containing source language text
output_file = 'small_dataset/data/multi30k-en-de/train_src.de'  # Path to the output file for the translated text

src_lines = read_from_file(input_file)
translated_lines = [translate(line.strip(), src_lang, tgt_lang) + '\n' for line in src_lines]

write_to_file(output_file, translated_lines)
print(f"Translation complete. Translated text saved to {output_file}")

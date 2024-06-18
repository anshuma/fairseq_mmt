import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sacrebleu

model_checkpoint = "t5-large"  # Using T5 large model for better performance
# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# Ensure the model is in evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(text, src_lang, tgt_lang):
    # Prepare the input text
    input_text = f"translate {src_lang} to {tgt_lang}: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def read_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

src_lang = 'English'
tgt_lang = 'French'

#input_file = 'small_dataset/data/multi30k-en-de/train.en'  # Path to the input file containing source language text
#output_file = 'small_dataset/data/multi30k-en-de/train_src.de'  # Path to the output file for the translated text
#reference_file = 'small_dataset/data/multi30k-en-de/train.de'  # Path to the reference file containing target language text

input_file = 'dataset/data/task1/raw/train.en'  # Path to the input file containing source language text
output_file = 'dataset/data/task1/raw/train_T5.fr'  # Path to the output file for the translated text
reference_file = 'dataset/data/task1/raw/train.fr'  # Path to the reference file containing target language text
# Read source lines and reference lines
src_lines = read_from_file(input_file)
ref_lines = read_from_file(reference_file)

# Translate source lines
translated_lines = [translate(line.strip(), src_lang, tgt_lang) + '\n' for line in src_lines]

# Write translations to file
write_to_file(output_file, translated_lines)
print(f"Translation complete. Translated text saved to {output_file}")

# Calculate BLEU score
translated_lines_stripped = [line.strip() for line in translated_lines]
ref_lines_stripped = [line.strip() for line in ref_lines]
bleu = sacrebleu.corpus_bleu(translated_lines_stripped, [ref_lines_stripped])

print(f"BLEU score: {bleu.score:.2f}")

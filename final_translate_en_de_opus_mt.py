import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

model_checkpoint = "Helsinki-NLP/opus-mt-en-de"
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Ensure the model is in evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(text, src_lang, tgt_lang):
    # Prepare the input text
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def read_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]
        #return file.readlines()

def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

src_lang = 'en'
tgt_lang = 'deu'

input_file = 'raw_data/test_2016_flickr.en'  # Path to the input file containing source language text
output_file = 'raw_data/test_2016_flickr_trans.de'  # Path to the output file for the translated text
reference_file = 'raw_data/test_2016_flickr.de'  # Path to the reference file containing target language text
#input_file = 'data/multi30k-en-de/test.2016.en'  # Path to the input file containing source language text
#output_file = 'data/multi30k-en-de/test.2016trans.de'  # Path to the output file for the translated text
#reference_file = 'data/multi30k-en-de/test.2016.de'  # Path to the reference file containing target language text

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

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import sacrebleu

model_checkpoint = "facebook/m2m100_418M"
# Load the tokenizer and model
tokenizer = M2M100Tokenizer.from_pretrained(model_checkpoint)
model = M2M100ForConditionalGeneration.from_pretrained(model_checkpoint)

# Ensure the model is in evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(text, src_lang, tgt_lang):
    # Prepare the input text
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def read_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

src_lang = 'en'
tgt_lang = 'de'

input_file = 'raw_data/test.2016.en'  # Path to the input file containing source language text
output_file = 'raw_data/test.2016_M2M100_src_de'  # Path to the output file for the translated text
reference_file = 'raw_data/test.2016.de'  # Path to the reference file containing target language text

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

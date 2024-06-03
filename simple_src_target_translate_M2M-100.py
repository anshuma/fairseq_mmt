from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def translate_text(model, tokenizer, text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# Load the model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Load source and reference texts
source_texts = load_texts('small_dataset/data/multi30k-en-de/train.en')
reference_texts = [line.split('\t') for line in load_texts('small_dataset/data/multi30k-en-de/train.de')]
output_file = 'small_dataset/data/multi30k-en-de/train_src_M2M.de'

# Define source and target languages
src_lang = "en"  # English
tgt_lang = "de"  # German

# Generate translations
translated_texts = [translate_text(model, tokenizer, text, src_lang, tgt_lang) for text in source_texts]
write_to_file(output_file, translated_texts)
# Calculate BLEU-4 score
smoothie = SmoothingFunction().method4
bleu_scores = [
    sentence_bleu([ref], trans.split(), smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
    for ref, trans in zip(reference_texts, translated_texts)
]

avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"BLEU-4 Score: {avg_bleu_score:.4f}")
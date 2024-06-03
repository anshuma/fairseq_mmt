from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def translate_text(model, tokenizer, text):
    input_text = f"translate English to German: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    outputs = model.generate(inputs.input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# Load the model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load source and reference texts
source_texts = load_texts('small_dataset/data/multi30k-en-de/train.en')
reference_texts = [line.split('\t') for line in load_texts('small_dataset/data/multi30k-en-de/train.de')]
output_file = 'small_dataset/data/multi30k-en-de/train_src_T5.de'
# Generate translations
translated_texts = [translate_text(model, tokenizer, text) for text in source_texts]

# Calculate BLEU-4 score
smoothie = SmoothingFunction().method4
bleu_scores = [
    sentence_bleu([ref], trans.split(), smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
    for ref, trans in zip(reference_texts, translated_texts)
]

avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"BLEU-4 Score: {avg_bleu_score:.4f}")

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load the fine-tuned model and tokenizer
model_name = "./finetuned_model_T5"
#model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure the model is in evaluation mode
# Ensure the model is in evaluation mode
model.eval()


def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def generate_translation(text, model, tokenizer,device):
    # Prepare the input text
    input_text = f"translate English to German: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Generate the translation
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


def remove_bpe(text):
    return text.replace("@@ ", "").replace("@@", "")


def calculate_bleu_score(reference_texts, translated_texts):
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
        for ref, pred in zip(reference_texts, translated_texts)
    ]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score


# Load new texts and reference texts from files
new_texts = load_texts('data/multi30k-en-de/train.en')
reference_texts = load_texts('data/multi30k-en-de/train.de')
reference_texts = [remove_bpe(text) for text in reference_texts]

# Generate translations for all new texts
translated_texts = [generate_translation(text, model, tokenizer, device) for text in new_texts]

# Remove BPE tokens from the translations
translated_texts = [remove_bpe(text) for text in translated_texts]

# Calculate the BLEU-4 score
bleu_score = calculate_bleu_score(reference_texts, translated_texts)
print(f"BLEU-4 score: {bleu_score}")

# Optionally, save the translated texts to a file
with open('data/multi30k-en-de/train_T5.de', 'w', encoding='utf-8') as file:
    for text in translated_texts:
        file.write(text + '\n')

import os
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load and prepare the dataset
def load_texts(source_path, target_path):
    with open(source_path, 'r', encoding='utf-8') as src_file:
        source_texts = [line.strip() for line in src_file.readlines()]
    with open(target_path, 'r', encoding='utf-8') as tgt_file:
        target_texts = [line.strip() for line in tgt_file.readlines()]
    return source_texts, target_texts

def create_dataset(source_texts, target_texts):
    return Dataset.from_dict({"translation": [{"src": src, "tgt": tgt} for src, tgt in zip(source_texts, target_texts)]})

#train_source_texts, train_target_texts = load_texts('small_dataset/data/multi30k-en-de/train.en', 'small_dataset/data/multi30k-en-de/train.de')
#val_source_texts, val_target_texts = load_texts('small_dataset/data/multi30k-en-de/valid.en', 'small_dataset/data/multi30k-en-de/valid.de')
train_source_texts, train_target_texts = load_texts('data/multi30k-en-de/train.en', 'data/multi30k-en-de/train.de')
val_source_texts, val_target_texts = load_texts('data/multi30k-en-de/valid.en', 'data/multi30k-en-de/valid.de')

train_dataset = create_dataset(train_source_texts, train_target_texts)
val_dataset = create_dataset(val_source_texts, val_target_texts)

datasets = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Load the tokenizer and model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocess the data
def preprocess_function(examples):
    inputs = [ex['src'] for ex in examples["translation"]]
    targets = [ex['tgt'] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=10_000,
)

# Define a custom compute metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    decoded_labels = [[label.replace('<pad>', '').strip()] for label in decoded_labels]

    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], pred.split(), smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
        for ref, pred in zip(decoded_labels, decoded_preds)
    ]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return {"bleu": avg_bleu_score}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
#trainer.save_model("./finetuned_model")
#tokenizer.save_pretrained("./finetuned_model")
trainer.save_model("./finetuned_model_NLLB")
tokenizer.save_pretrained("./finetuned_model_NLLB")

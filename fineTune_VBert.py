# import torch
# from transformers import VisualBertForQuestionAnswering, BertTokenizer, Trainer, TrainingArguments
# from datasets import Dataset, Features, Value, Array3D
# from PIL import Image
# from torchvision.transforms import ToTensor
# import pandas as pd
# import os
#
# # Define paths
# data_dir = '/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/data/multi30k-en-de'
# image_dir = '/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/flickr30k-images/'
# image_idx_dir = '/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/'
#
# def get_filenames(path):
#     l = []
#     with open(path, 'r') as f:
#         for line in f:
#             l.append(line.strip().split('#')[0])
#     return l
# # Load indices and captions
# def load_data(split):
#     #print(f'{image_dir}/{split}.txt')
#     #print(f'{data_dir}/{split}.en')
#     #indices = pd.read_csv(f'{image_dir}/{split}.txt', header=None, names=['index'])
#     #captions = pd.read_csv(f'{data_dir}/{split}.en', header=None, names=['caption'])
#     #indices = get_filenames(os.path.join(f'{image_dir}/{split}.txt'))
#     #data = pd.concat([indices, captions], axis=1)
#     #return data
#     try:
#         with open(f'{image_idx_dir}/{split}.txt', 'r') as f:
#             indices = f.read().splitlines()
#         with open(f'{data_dir}/{split}.en', 'r') as f:
#             captions = f.read().splitlines()
#         data = pd.DataFrame({'index': indices, 'caption': captions})
#         return data
#     except Exception as e:
#         print(f"Error loading {split} data: {e}")
#         return None
#
#
# # Preprocess function
# def preprocess_data(row):
#     image_path = os.path.join(image_dir, f"{row['index']}")
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image)
#
#     encoding = tokenizer(row['caption'], padding='max_length', truncation=True)
#     encoding['pixel_values'] = image
#     encoding['labels'] = 0  # Placeholder for the label
#     return encoding
#
#
# # Initialize tokenizer and transform
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# transform = ToTensor()
#
# # Load and preprocess datasets
# train_data = load_data('train')
# valid_data = load_data('valid')
# test_data = load_data('test.2016')
#
# if train_data is not None:
#     train_encodings = train_data.apply(preprocess_data, axis=1).tolist()
# if valid_data is not None:
#     valid_encodings = valid_data.apply(preprocess_data, axis=1).tolist()
# if test_data is not None:
#     test_encodings = test_data.apply(preprocess_data, axis=1).tolist()
#
# # Convert to Hugging Face Dataset
# features = Features({
#     'input_ids': Value(dtype='int32'),
#     'attention_mask': Value(dtype='int32'),
#     'pixel_values': Array3D(dtype='float32', shape=(3, 224, 224)),
#     'labels': Value(dtype='int32')
# })
#
# def create_dataset(encodings):
#     return Dataset.from_dict({key: [d[key] for d in encodings] for key in encodings[0]})
#
# train_dataset = create_dataset(train_encodings) if train_data is not None else None
# valid_dataset = create_dataset(valid_encodings) if valid_data is not None else None
# test_dataset = create_dataset(test_encodings) if test_data is not None else None
#
#
# # Initialize the model
# model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
#
# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
# )
#
# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
# )
#
# # Fine-tune the model
# trainer.train()
#
# # Evaluate the model
# trainer.evaluate(eval_dataset=test_dataset)


import torch
from transformers import VisualBertForQuestionAnswering, BertTokenizer, Trainer, TrainingArguments
from transformers import AutoImageProcessor, DeiTModel
from datasets import Dataset, Features, Value, Array3D
from PIL import Image
from torchvision.transforms import ToTensor
import pandas as pd
import os


# Define paths
data_dir = '/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/data/multi30k-en-de'
image_dir = '/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/flickr30k-images/'
image_idx_dir = '/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/'
count = 0
# Load indices and captions
def load_data(split):
    try:
        with open(f'{image_idx_dir}/{split}.txt', 'r') as f:
            indices = f.read().splitlines()
        with open(f'{data_dir}/{split}.en', 'r') as f:
            captions = f.read().splitlines()
        data = pd.DataFrame({'index': indices, 'caption': captions})
        return data
    except Exception as e:
        print(f"Error loading {split} data: {e}")
        return None


image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
image_model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
def get_visual_embeddings(image):
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = image_model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    list(last_hidden_states.shape)
    return last_hidden_states

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Preprocess function
def preprocess_data(row):
    image_path = os.path.join(image_dir, f"{row['index']}")
    image = Image.open(image_path).convert('RGB')
    #visual_embeds = get_visual_embeddings(image).unsqueeze(0)
    visual_embeds = get_visual_embeddings(image)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    inputs = tokenizer(row['caption'], padding='max_length', truncation=True, return_tensors='pt')
    #input_ids = torch.tensor(tokens["input_ids"])
    #attention_mask = torch.tensor(tokens["attention_mask"])
    #token_type_ids = torch.tensor(tokens["token_type_ids"])
    inputs.update({
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    })
    #print(inputs)
    global count
    if(count == 0):
        print('visual_embeds',visual_embeds.shape)
        print('visual_attention_mask.shape',visual_attention_mask.shape)
        print('attention_mask.shape', inputs['attention_mask'].shape)
        count = count+1
    return {key: value.squeeze(0) for key, value in inputs.items()}



# Load and preprocess datasets
train_data = load_data('train')
valid_data = load_data('valid')
test_data = load_data('test.2016')

if train_data is not None:
    train_encodings = train_data.apply(preprocess_data, axis=1).tolist()
if valid_data is not None:
    valid_encodings = valid_data.apply(preprocess_data, axis=1).tolist()
if test_data is not None:
    test_encodings = test_data.apply(preprocess_data, axis=1).tolist()

# Convert to Hugging Face Dataset
features = Features({
    'input_ids': Value(dtype='int32'),
    'attention_mask': Value(dtype='int32'),
    'token_type_ids': Value(dtype='int32'),
    'visual_embeds': Array3D(dtype='float32', shape=(3, 224, 224)),
    'visual_token_type_ids': Value(dtype='int32'),
    'visual_attention_mask': Value(dtype='float32')
})


def create_dataset(encodings):
    dict_data = {key: [d[key].tolist() for d in encodings] for key in encodings[0]}
    return Dataset.from_dict(dict_data)


train_dataset = create_dataset(train_encodings) if train_data is not None else None
valid_dataset = create_dataset(valid_encodings) if valid_data is not None else None
test_dataset = create_dataset(test_encodings) if test_data is not None else None

# Initialize the model
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Fine-tune the model
if train_dataset is not None:
    trainer.train()

# Evaluate the model
if test_dataset is not None:
    trainer.evaluate(eval_dataset=test_dataset)

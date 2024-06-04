import torch
from transformers import BertTokenizer, VisualBertForPreTraining, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pandas as pd
import os

# Define device as CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and model components
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
blip_model.eval()

# Define paths
data_dir = '../small_dataset/data/multi30k-en-de'
image_dir = '../small_dataset/flickr30k/flickr30k-images/'
image_idx_dir = '../flickr30k/'
output_dir = '../data/VisualBert_blip_large'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

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

# Linear projection with CUDA
linear_projection = torch.nn.Linear(1024, 2048).to(device)


def get_visual_embedding_blip(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=img, return_tensors="pt").to(device)
    vision_outputs = blip_model.vision_model(pixel_values=inputs['pixel_values'], return_dict=True)
    out = vision_outputs.last_hidden_state
    out = linear_projection(out)
    return out


# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device)


def preprocess_example(image_path, text):
    visual_embeds = get_visual_embedding_blip(image_path)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float, device=device)

    inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt').to(device)
    labels = tokenizer(text, return_tensors="pt", padding="max_length",
                       max_length=inputs["input_ids"].shape[-1] + visual_embeds.shape[-2])["input_ids"].to(device)

    inputs.update({
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
        "labels": labels.squeeze(0)
    })

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1]


def load_and_preprocess_data(split):
    data = load_data(split)
    outputs = []
    if data is not None:
        for index, row in data.iterrows():
            inputs = preprocess_example(os.path.join(image_dir, f"{row['index']}"), row['caption'])
            outputs.append(inputs)
    return outputs


def generate_and_save_predictions(encodings, split):
    global max_length
    tmp = []
    tmp1 = []
    global count
    model.eval()
    predictions = []
    for inputs in encodings:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            prediction_logits = outputs.prediction_logits
            predicted_tokens = torch.argmax(prediction_logits, dim=-1)
            predicted_sentence = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
            predictions.append(predicted_sentence)
            last_layer_output = outputs.hidden_states[-1]
            print('last_layer_output', last_layer_output.shape, flush=True)
            tmp.append(last_layer_output.detach().to('device'))
            if len(tmp) == 2000:
                res = torch.cat(tmp)
                print(res.shape, flush=True)
                torch.save(res, os.path.join(output_dir, str(count) + split + '.pth'))
                count += 1
                tmp = []

    with open(os.path.join(output_dir, f"{split}_predictions.txt"), 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    print('tmp', tmp, flush=True)
    res = torch.cat(tmp).cpu()
    if count > 1:
        torch.save(res, os.path.join(output_dir, 'final' + split + '.pth'))
    else:
        print('feature shape:', res.shape, ',save in:', output_dir + '/' + split + '.pth', flush=True)
        torch.save(res, os.path.join(output_dir, split + '.pth'))

    del tmp
    _tmp = []
    if count > 1:
        for i in range(1, count):
            _tmp.append(torch.load(os.path.join(output_dir, str(i) + split + '.pth')))
        _tmp.append(torch.load(os.path.join(output_dir, 'final' + split + '.pth')))
        res = torch.cat(_tmp).cpu()
        print('feature shape:', res.shape, ',save in:', output_dir + '/' + split + '.pth', flush=True)
        torch.save(res, os.path.join(output_dir, split + '.pth'))

        # delete
        for i in range(1, count):
            os.remove(os.path.join(output_dir, str(i) + split + '.pth'))
        os.remove(os.path.join(output_dir, 'final' + split + '.pth'))


# Similar modification for prediction function to utilize CUDA

# Process and generate outputs for train, validation, and test sets
train_outputs = load_and_preprocess_data('train')
generate_and_save_predictions(train_outputs, 'train')

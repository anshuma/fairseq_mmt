import torch
from transformers import  BertTokenizer, VisualBertForPreTraining
from PIL import Image
import pandas as pd
import os
from transformers import BlipProcessor, BlipForConditionalGeneration


# load models and model components
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model.eval()

# Define paths
data_dir = '../small_dataset/data/multi30k-en-de'
image_dir = '../small_dataset/flickr30k/flickr30k-images/'
image_idx_dir = '../flickr30k/'
count1 = 0
output_dir = '../data/VisualBert_blip_large'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
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


linear_projection = torch.nn.Linear(1024, 2048)
def get_visual_embedding_blip(image_path):
    #i = os.path.join(flickr30k_path, dic[dataset] + '-images', i)
    img = Image.open(image_path).convert("RGB")
    # input = transform(img).unsqueeze(0).to('cuda:0') # transform and add batch dimension
    inputs = blip_processor(images=img, return_tensors="pt")#.to(device)
    vision_outputs = blip_model.vision_model(pixel_values=inputs['pixel_values'], return_dict=True)
    out = vision_outputs.last_hidden_state
    out = linear_projection(out)
    return out

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
# Preprocess function
max_length = 0
def preprocess_example(image_path, text):
    global max_length
    image_path = os.path.join(image_path)
    #image = Image.open(image_path).convert('RGB')
    #visual_embeds = get_visual_embeddings(image).unsqueeze(0)
    #visual_embeds = get_visual_embeddings1([image_path])
    visual_embeds = get_visual_embedding_blip(image_path)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
    max_length = inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]
    #print('max_length',max_length)
    labels = tokenizer(text, return_tensors="pt", padding="max_length", max_length=max_length)[
        "input_ids"]
    #print('labels.shape',labels.shape)
    sentence_image_labels = torch.tensor(1).unsqueeze(0)
    inputs.update({
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
        "labels":labels.squeeze(0),
        "sentence_image_labels":sentence_image_labels
    })
    global count1
    if(count1 == 0):
        print('visual_embeds',visual_embeds.shape)
        print('visual_attention_mask.shape',visual_attention_mask.shape)
        print('attention_mask.shape', inputs['attention_mask'].shape)
    count1 = count1+1
    print('count', count1)
    if(count1 % 50 == 0):
        print('image_path',image_path)
        print('count1',count1)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_layer_output = outputs.hidden_states[-1]
    return last_layer_output
def load_and_preprocess_data(split):
    data = load_data(split)
    outputs = []
    if data is not None:
        for index, row in data.iterrows():
            inputs = preprocess_example(os.path.join(image_dir, f"{row['index']}"), row['caption'])
            outputs.append(inputs)
    return outputs

model = VisualBertForPreTraining.from_pretrained('./finetuned_model_VisualBERT_small')
count = 1
def generate_and_save_predictions(outputs, split):
    global max_length
    tmp = []
    tmp1 = []
    global count
    model.eval()
    predictions = []
    for last_layer_output in outputs:
            print('last_layer_output', last_layer_output.shape)
            tmp.append(last_layer_output.detach())
            if len(tmp) == 2000:
                res = torch.cat(tmp)
                print(res.shape)
                torch.save(res, os.path.join(output_dir, str(count) + split + '.pth'))
                count += 1
                tmp = []

    print('tmp', tmp)
    res = torch.cat(tmp)
    if count > 1:
        torch.save(res, os.path.join(output_dir, 'final' + split + '.pth'))
    else:
        print('feature shape:', res.shape, ',save in:', output_dir + '/' + split + '.pth')
        torch.save(res, os.path.join(output_dir, split + '.pth'))

    del tmp
    _tmp = []
    if count > 1:
        for i in range(1, count):
            _tmp.append(torch.load(os.path.join(output_dir, str(i) + split + '.pth')))
        _tmp.append(torch.load(os.path.join(output_dir, 'final' + split + '.pth')))
        res = torch.cat(_tmp).cpu()
        print('feature shape:', res.shape, ',save in:', output_dir + '/' + split + '.pth')
        torch.save(res, os.path.join(output_dir, split + '.pth'))

        # delete
        for i in range(1, count):
            os.remove(os.path.join(output_dir, str(i) + split + '.pth'))
        os.remove(os.path.join(output_dir, 'final' + split + '.pth'))

# Process and generate outputs for train, validation, and test sets
train_outputs = load_and_preprocess_data('train')
#valid_outputs = load_and_preprocess_data('valid')
#test_outputs = load_and_preprocess_data('test.2016')

generate_and_save_predictions(train_outputs, 'train')
#generate_and_save_predictions(valid_outputs, 'valid')
#generate_and_save_predictions(test_outputs, 'test.2016')
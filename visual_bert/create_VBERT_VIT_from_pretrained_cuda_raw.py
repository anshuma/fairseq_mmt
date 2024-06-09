import torch
from transformers import BertTokenizer, VisualBertForPreTraining, BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
import timm
# Load the processor and model
from PIL import Image
import pandas as pd
import os
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
torch.cuda.empty_cache()
# Define device as CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and model components
#blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
#blip_model.eval()

#vit_processor = AutoProcessor.from_pretrained("vit_base_patch14_reg4_dinov2.lvd142m")
#vit_model = AutoModel.from_pretrained("vit_base_patch14_reg4_dinov2.lvd142m").to(device)
vit_model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m',pretrained=True,num_classes=0,).to(device)
vit_model.eval()

config = resolve_data_config({}, model=vit_model)
transform = create_transform(**config)
# Define paths
data_dir = '../data/multi30k-en-de/'
image_dir = '../flickr30k/flickr30k-images/'
image_idx_dir = '../flickr30k/'
output_dir = '../data/VisualBert_blip_large_DE_8June'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_data(split):
    try:
        with open(f'{image_idx_dir}/{split}.txt', 'r') as f:
            indices = f.read().splitlines()
        with open(f'{data_dir}/{split}.de', 'r') as f:
            captions = f.read().splitlines()
        with open(f'{data_dir}/{split}_T5.de', 'r') as f:
            captions1 = f.read().splitlines()
        data = pd.DataFrame({'index': indices, 'caption': captions})
        return data
    except Exception as e:
        print(f"Error loading {split} data: {e}")
        return None

# Linear projection with CUDA
linear_projection = torch.nn.Linear(1024, 2048).to(device)


def get_visual_embedding_blip(image_path):
    img = Image.open(image_path).convert("RGB")
    #inputs = vit_processor(images=img, return_tensors="pt").to(device)
    #vision_outputs = vit_model(pixel_values=inputs['pixel_values'], return_dict=True)
    out = vit_model.forward_features(transform(img).unsqueeze(0).to(device))
    #out = vision_outputs.last_hidden_state
    print('out.shape',out.shape)
    out = linear_projection(out)
    return out


# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device)

#model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

mapping_output = torch.nn.Linear(1089, 768).to(device)
count1 = 0
def load_and_preprocess_data(split):
    data = load_data(split)
    global max_length
    tmp = []
    tmp1 = []
    global count1
    count = 1
    global device
    model.eval()
    predictions = []
    if data is not None:
        with torch.no_grad():
            for index, row in data.iterrows():
                text1 = row['caption']
                text2 = row['caption1']
                image_path = os.path.join(image_dir, f"{row['index']}")

                #image = Image.open(image_path)
                visual_embeds = get_visual_embedding_blip(image_path)
                visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float, device=device)

                inputs = tokenizer([text1,text2], padding='max_length', truncation=True, return_tensors='pt').to(device)
                labels = tokenizer([text1,text2], return_tensors="pt", padding="max_length",
                                   max_length=inputs["input_ids"].shape[-1] + visual_embeds.shape[-2])["input_ids"].to(
                    device)
                #if visual_attention_mask.dim() == 1:
                #    visual_attention_mask = visual_attention_mask.unsqueeze(0)
                #    visual_embeds = visual_embeds.unsqueeze(0)
                # Adjust visual_attention_mask to match the batch size of attention_mask
                #visual_attention_mask = visual_attention_mask.expand(inputs['attention_mask'].size(0), -1)
                #visual_embeds = visual_embeds.expand(inputs['attention_mask'].size(0), -1)
                
                inputs.update({
                    "visual_embeds": [visual_embeds,visual_embeds],
                    "visual_token_type_ids": [visual_token_type_ids,visual_token_type_ids],
                    "visual_attention_mask": [visual_attention_mask,visual_attention_mask],
                    "labels": labels.squeeze(0)
                })
                global count1
                if (count1 == 0):
                    print('visual_embeds', visual_embeds.shape, flush=True)
                    print('visual_attention_mask.shape', visual_attention_mask.shape, flush=True)
                    print('attention_mask.shape', inputs['attention_mask'].shape, flush=True)
                    print('attention_mask.shape', inputs['attention_mask'].shape, flush=True)
                count1 = count1 + 1
                print('count', count1, flush=True)
                if (count1 % 50 == 0):
                    print('image_path', image_path, flush=True)
                    print('count1', count1, flush=True)
                '''
                #last_layer_output = outputs.hidden_states[-1]
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # Forward pass through the model to get the last hidden state
                with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                '''
                outputs = model(**inputs, output_hidden_states=True)
                last_layer_output = outputs.hidden_states[-1]

                if (count1 == 1):
                    # Print the shape of the last hidden state
                    print("last_layer_output shape:", last_layer_output.shape, flush=True)
                '''
                # Optional: If you want to get the logits and probabilities as in the original example
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

                if (count1 == 1):
                    print("Logits per imagei shape:", logits_per_image.shape,flush=True)
                    print("Probabilities:", probs.shape, flush=True)
                '''
                #last_layer_output = last_layer_output.squeeze(0)
                #last_layer_output = mapping_output(last_layer_output)
                #last_layer_output = last_layer_output.unsqueeze(0)
                #tmp.append(last_layer_output.detach().to('device'))
                print('last_layer_output', last_layer_output.shape, flush=True)
                M, N, O = last_layer_output.shape
                if N > O:
                    last_layer_output= last_layer_output[:, :O, :]
                else:
                    last_layer_output = last_layer_output[:, :, :N]
                 #tmp.append(last_layer_output.detach().to('device'))
                print('last_layer_output', last_layer_output.shape, flush=True)
                tmp.append(last_layer_output.detach().to(device))
                if len(tmp) == 2000:
                    res = torch.cat(tmp)
                    print(res.shape, flush=True)
                    torch.save(res, os.path.join(output_dir, str(count) + split + '.pth'))
                    count += 1
                    tmp = []

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



# Process and generate outputs for train, validation, and test sets
train_outputs = load_and_preprocess_data('train')
#generate_and_save_predictions(train_outputs, 'train')

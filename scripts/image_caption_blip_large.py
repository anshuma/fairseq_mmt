import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Load the processor and model
#processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# Move model to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVision2Seq
#processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
#model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

#processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")


#processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
#model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)



def generate_caption(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    #inputs = {k: v.to(device) for k, v in inputs.items()}
    #model.to(device)

    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        #outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)

    # Decode the generated text
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def read_image_list(file_path):
    with open(file_path, 'r') as file:
        image_files = file.read().splitlines()
    return image_files

def generate_captions_for_images(image_list, image_folder, output_file):
    captions = []
    count = 0
    for image_file in image_list:
        image_path = os.path.join(image_folder, image_file)
        if os.path.exists(image_path):
            caption = generate_caption(image_path)
            captions.append(f"{caption}")
            print(f"{caption}")
        else:
            print(f"Image {image_file} not found in {image_folder}")
    
    with open(output_file, 'w') as file:
        for caption in captions:
            file.write(caption + "\n")

# Example usage
#train_txt_path = 'small_dataset/flickr30k/train.txt'  # Path to the train.txt file
train_txt_path = 'small_dataset/flickr30k/test_2016_flickr.txt'  # Path to the train.txt file
image_folder = 'small_dataset/flickr30k/flickr30k-images'  # Path to the folder containing .jpg files
#output_file = 'small_dataset/flickr30k/train_captions.en'  # Output file to save the captions
output_file = 'small_dataset/flickr30k/test2016_captions.en'  # Output file to save the captions

# Read the list of image files
image_list = read_image_list(train_txt_path)

# Generate captions for the images and save them to the output file
generate_captions_for_images(image_list, image_folder, output_file)
print(f"Captions saved to {output_file}")

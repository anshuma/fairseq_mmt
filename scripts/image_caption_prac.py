
from transformers import pipeline
from PIL import Image
import requests

model_id = "xtuner/llava-phi-3-mini-hf"
pipe = pipeline("image-to-text", model=model_id, device=0)
#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"

#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("small_dataset/flickr30k/flickr30k-images/1000092795.jpg").convert("RGB")
#prompt = "<|user|>\n<image>\nWhat does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud<|end|>\n<|assistant|>\n"

outputs = pipe(image, generate_kwargs={"max_new_tokens": 200})
print(outputs)
#>>> [{'generated_text': '\nWhat does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud (1) lava'}]


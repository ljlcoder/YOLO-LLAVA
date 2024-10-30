from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("./llava_model",device_map='cuda:0')
processor = AutoProcessor.from_pretrained("./llava_model")

prompt = "USER: <image>\nHow many people in the image? ASSISTANT:"
url = "person.jpg"
image = Image.open(url)

inputs = processor(images=image, text=prompt, return_tensors="pt")
for item_key in inputs.keys():
    inputs[item_key] = inputs[item_key].to('cuda:0')
generate_ids = model.generate(**inputs, max_new_tokens=15)
result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(result)
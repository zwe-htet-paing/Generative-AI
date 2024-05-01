import os
import io
import base64
import requests
import json
from PIL import Image
from text_generation import Client


class APIHandler:
    def __init__(self, HUGGINGFACE_API_KEY):
        self.headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        self.summarization_endpoint = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
        self.ner_endpoint = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
        self.image_captioning_endpoint = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
        self.image_generation_endpoint = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        self.chat_endpoint = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

        # FalcomLM-instruct endpoint on the text_generation library
        self.client = Client(self.chat_endpoint, headers=self.headers, timeout=120)


    def get_completion(self, inputs, parameters=None, ENDPOINT_URL=None, task=None):
        payload = {"inputs": inputs}
        if parameters is not None:
            payload.update({"parameters": parameters})

        while True:
            try:
                response = requests.post(ENDPOINT_URL,
                                        headers=self.headers,
                                        json=payload)
                break
            except Exception:
                continue

        if task == 'image-generation':
            return response.content
        return response.json()
    

    def summarize(self, inputs, parameters=None):
        output = self.get_completion(inputs, parameters=parameters, ENDPOINT_URL=self.summarization_endpoint)
        return output[0]["summary_text"]
    

    def ner(self, inputs, parameters=None):
        output = self.get_completion(inputs, parameters=parameters, ENDPOINT_URL=self.ner_endpoint)
        return {"text": inputs, "entities": output}
    

    def image_captioning(self, image, parameters=None):
        base64_image = self.image_to_base64(image)
        output = self.get_completion(inputs=base64_image, parameters=parameters, ENDPOINT_URL=self.image_captioning_endpoint)
        return output[0]['generated_text']
    
    def image_generation(self, prompt, parameters=None):
        image_bytes =self.get_completion(inputs=prompt, parameters=parameters, ENDPOINT_URL=self.image_generation_endpoint, task="image-generation")
        return self.base64_to_pil(image_bytes)
    
    def image_generation_v2(self, prompt, negative_prompt, steps, guidance, width, height):
        params = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height
        }
        
        output = self.get_completion(prompt, params, ENDPOINT_URL=self.image_generation_endpoint, task="image-generation")
        pil_image = self.base64_to_pil(output)
        return pil_image

    @staticmethod
    def image_to_base64(PILImage):
        # Convert the image to bytes
        byte_arr = io.BytesIO()
        PILImage.save(byte_arr, format="PNG")
        byte_arr = byte_arr.getvalue()

        # Encode the bytes image to base64
        base64_img = base64.b64encode(byte_arr)

        # Convert bytes to string
        base64_img_str = base64_img.decode('utf-8')
        return str(base64_img_str)
    
    @staticmethod
    def base64_to_pil(image_bytes):
        # image_bytes = base64.b64decode(image_bytes)
        pil_image = Image.open(io.BytesIO(image_bytes))
        return pil_image
    
    def chat_with_LLM(self, inputs, slider):
        output = self.client.generate(inputs, max_new_tokens=slider).generated_text
        return output
    
    




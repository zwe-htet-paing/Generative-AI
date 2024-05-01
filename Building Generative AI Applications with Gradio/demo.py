import os
from dotenv import load_dotenv
from api_handler import APIHandler
import matplotlib.pyplot as plt

import gradio as gr
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# print(API_KEY)
# print(type(API_KEY))


class TestAPI:
    def __init__(self):
        self.apis = APIHandler(HUGGINGFACE_API_KEY=API_KEY)

    def test_summarize(self):
        text = '''The tower is 324 metres (1,063 ft) tall, about the same height
                as an 81-storey building, and the tallest structure in Paris. 
                Its base is square, measuring 125 metres (410 ft) on each side. 
                During its construction, the Eiffel Tower surpassed the Washington 
                Monument to become the tallest man-made structure in the world,
                a title it held for 41 years until the Chrysler Building
                in New York City was finished in 1930. It was the first structure 
                to reach a height of 300 metres. Due to the addition of a broadcasting 
                aerial at the top of the tower in 1957, it is now taller than the 
                Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the 
                Eiffel Tower is the second tallest free-standing structure in France 
                after the Millau Viaduct.'''
        output = self.apis.summarize(text)
        print(output)


    def test_ner(self):
        text = "My name is Oliver Zwe, I'm building DeeplearningAI and I live in California"
        output = self.apis.ner(text)
        print(output)

    def test_image_captioning(self):
        image_path = "samples/images/pennsylvania.jpg"
        image = Image.open(image_path)
        output = self.apis.image_captioning(image)
        print(output)

    def test_image_generation(self):
        prompt = "a dog running in a park"
        image = self.apis.image_generation(prompt)

        plt.imshow(image)
        plt.axis('off')
        plt.show()


class Demo:
    def __init__(self):
        self.apis = APIHandler(HUGGINGFACE_API_KEY=API_KEY)

    def launch(self):
        summarization_demo = gr.Interface(
            fn=self.apis.summarize,
            inputs=[gr.Textbox(label="Text to summarize", lines=6)],
            outputs=[gr.Textbox(label="Result", lines=3)],
            title="Text summarizatoin using Falconai",
            description="Summarize any text using the `Falconsai/text_summarization` model under the hood!",
            allow_flagging="never",
            examples=['''The tower is 324 metres (1,063 ft) tall, about the same height
                    as an 81-storey building, and the tallest structure in Paris. 
                    Its base is square, measuring 125 metres (410 ft) on each side. 
                    During its construction, the Eiffel Tower surpassed the Washington 
                    Monument to become the tallest man-made structure in the world,
                    a title it held for 41 years until the Chrysler Building
                    in New York City was finished in 1930. It was the first structure 
                    to reach a height of 300 metres. Due to the addition of a broadcasting 
                    aerial at the top of the tower in 1957, it is now taller than the 
                    Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the 
                    Eiffel Tower is the second tallest free-standing structure in France 
                    after the Millau Viaduct.''']
            )

        ner_demo = gr.Interface(
            fn=self.apis.ner,
            inputs=[gr.Textbox(label="Text to find entities", lines=2)],
            outputs=[gr.HighlightedText(label="Text with entities")],
            title="NER with bert-base-NER",
            description="Find entities using the `dslim/bert-base-NER` model under the hood!",
            allow_flagging="never",
            examples=["My name is Oliver and I live in California and work at Meta", "My name is Poli and work at HuggingFace"]
            )

        image_captioning_demo = gr.Interface(
            fn=self.apis.image_captioning,
            inputs=[gr.Image(label="Upload image", type="pil")],
            outputs=[gr.Textbox(label="Caption")],
            title="Image Captioning with BLIP",
            description="Caption any image using `Salesforce/blip-image-captioning-base` under the hood!",
            allow_flagging="never",
            examples=['samples/images/pennsylvania.jpg']
            )

        image_generation_demo_v1 = gr.Interface(
            fn=self.apis.image_generation,
            inputs=[gr.Textbox(label="Prompt")],
            outputs=[gr.Image(label="Result")],
            title="Image Generation with Stable Diffusion",
            description="Generate any image using `runwayml/stable-diffusion-v1-5` under the hood!",
            allow_flagging="never",
            examples=["the spirit of a tamagotchi wandering in the city of Vienna","a mecha robot in a favela"]
            )
        

        image_generation_demo_v2 =  gr.Interface(
            fn=self.apis.image_generation_v2,
            inputs=[
                gr.Textbox(label="Your prompt"),
                gr.Textbox(label="Negative prompt"),
                gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                            info="In how many steps will the denoiser denoise the image?"),
                gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7, 
                            info="Controls how much the text prompt influences the result"),
                gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
                gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512),
            ],
            outputs=[gr.Image(label="Result")],
            title="Image Generation with Stable Diffusion",
            description="Generate any image using `runwayml/stable-diffusion-v1-5` under the hood!",
            allow_flagging="never",
            )

        chat_demo = gr.Interface(
            fn=self.apis.chat_with_LLM, 
            inputs=[gr.Textbox(label="Prompt"), 
                    gr.Slider(label="Max new tokens", 
                                value=20,  
                                maximum=1024, 
                                minimum=1)], 
            outputs=[gr.Textbox(label="Completion")],
            title="Chat with LLM",
            description="Chat with LLM using the `tiiuae/falcon-7b-instruct` under the hood!",
            allow_flagging="never",
            )

        demo = gr.Blocks()
        with demo:
            gr.TabbedInterface(
                [summarization_demo, ner_demo, image_captioning_demo, image_generation_demo_v2, chat_demo], 
                ["Summarization", "NER", "Image Captioning", "Image Generation","Chat With LLM"]
                )  
            
        demo.queue(max_size=10)
        demo.launch()

if __name__ == "__main__":
    # test = TestAPI()
    # test.test_ner()
    demo = Demo()
    demo.launch()

from PIL import Image
from io import BytesIO
import numpy as np
import base64
import torch
import torchvision.transforms.functional as F
from S2I import Sketch2Image
# from S2I.samer import SAMController


class Sketch2ImageController(Sketch2Image):
    def __init__(self, gr):
        super().__init__()
        self.gr = gr
        self.style_list = [
            {"name": "Cinematic",
             "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"},
            {"name": "3D Model",
             "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"},
            {"name": "Anime",
             "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed"},
            {"name": "Digital Art",
             "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"},
            {"name": "Photographic",
             "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"},
            {"name": "Pixel art", "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics"},
            {"name": "Fantasy art",
             "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy"},
            {"name": "Neonpunk",
             "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional"},
            {"name": "Manga",
             "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style"},
        ]

        self.styles = {k["name"]: k["prompt"] for k in self.style_list}
        self.STYLE_NAMES = list(self.styles.keys())
        self.DEFAULT_STYLE_NAME = "Fantasy art"
        self.MAX_SEED = np.iinfo(np.int32).max

        # Initialize the model once here
        # self.model = Sketch2Image()

    def update_canvas(self, use_line, use_eraser):
        brush_size = 20 if use_eraser else 4
        _color = "#ffffff" if use_eraser else "#000000"
        return self.gr.update(brush_radius=brush_size, brush_color=_color, interactive=True)

    def upload_sketch(self, file):
        _img = Image.open(file.name).convert("L")
        return self.gr.update(value=_img, source="upload", interactive=True)

    @staticmethod
    def pil_image_to_data_uri(img, format="PNG"):
        buffered = BytesIO()
        img.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"

    def artwork(self, image, prompt, prompt_template, style_name, seed, val_r, faster):

        # Handle case where both images are None
        if image is None:
            ones = Image.new("L", (512, 512), 255)
            temp_uri = self.pil_image_to_data_uri(ones)
            return ones, self.gr.update(link=temp_uri), self.gr.update(link=temp_uri)

        prompt = prompt_template.replace("{prompt}", prompt)
        image = image.convert("RGB")
        image_t = F.to_tensor(image) > 0.5

        # Start GPU operations
        c_t = image_t.unsqueeze(0).cuda().float()
        torch.manual_seed(seed)
        B, C, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)

        with torch.no_grad():
            output_image = self.generate(c_t, prompt, r=val_r, noise_map=noise, half_model=faster)

        output_pil = F.to_pil_image(output_image[0].cpu() * 0.5 + 0.5)
        input_segment_rgb = self.pil_image_to_data_uri(Image.fromarray(255 - np.array(image)))
        input_sketch_uri = self.pil_image_to_data_uri(Image.fromarray(255 - np.array(image)))
        output_image_uri = self.pil_image_to_data_uri(output_pil)

        return (
            output_pil,
            self.gr.update(link=input_segment_rgb),
            self.gr.update(link=input_sketch_uri),
            self.gr.update(link=output_image_uri),
        )

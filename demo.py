from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import requests
from io import BytesIO
from S2I import Sketch2Image
import gc


# example url
# url = 'https://i.pinimg.com/736x/45/ed/9a/45ed9ab804413847cc50d15cff80e0d4.jpg'
# url = 'https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/a6fbd156966637.59c34493302ee.png'
# url = 'https://cdn.vectorstock.com/i/500p/19/28/old-house-sketch-vector-471928.jpg'
# url = 'https://i.redd.it/06flvgjlt9y61.jpg'
# url = 'https://www.drawingskill.com/wp-content/uploads/5/Easy-Sketch-Art-Drawing.jpg'

# example prompt
# prompt = 'A beautiful Chinese house style with beautiful sky, with a sun'
# prompt = 'A beautiful city and crowded, high quality, sunny day'
# prompt = 'A beautiful England cat, high quality'
# prompt = 'A beutiful chinese temple with trees around'
def inference(url, prompt, gamma=0.5, seed=42, original_size=False):
    # initialize the model
    model = Sketch2Image()
    # make sure that the input image is a multiple of 8
    response = requests.get(url)
    input_image = Image.open(BytesIO(response.content)).convert('RGB')
    if not original_size:
        new_width = 512
        new_height = 512
    else:
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    with torch.no_grad():
        image_t = F.to_tensor(input_image) < 0.5
        c_t = image_t.unsqueeze(0).cuda().float()
        torch.manual_seed(seed)
        B, C, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
        output_image = model.generate(c_t, prompt, r=gamma, noise_map=noise)
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil.save('output.png')
        del c_t, noise, output_image
        torch.cuda.empty_cache()
        gc.collect()

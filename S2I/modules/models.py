import torch
from diffusers import DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel, AutoModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from S2I.modules.utils import sc_vae_encoder_fwd, sc_vae_decoder_fwd, download_url
import os


class PrimaryModel:
    def __init__(self, backbone_diffusion_path='stabilityai/sd-turbo'):
        self.backbone_diffusion_path = backbone_diffusion_path
        self.global_unet = None
        self.global_vae = None
        self.global_tokenizer = None
        self.global_text_encoder = None
        self.global_scheduler = None

    def one_step_scheduler(self, ):
        noise_scheduler_1step = DDPMScheduler.from_pretrained(self.backbone_diffusion_path, subfolder="scheduler")
        noise_scheduler_1step.set_timesteps(1, device="cuda")
        noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
        return noise_scheduler_1step

    @staticmethod
    def skip_connections(vae):
        vae.encoder.forward = sc_vae_encoder_fwd.__get__(vae.encoder,
                                                         vae.encoder.__class__)
        vae.decoder.forward = sc_vae_decoder_fwd.__get__(vae.decoder,
                                                         vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).cuda()
        vae.decoder.ignore_skip = False
        return vae

    def from_pretrained(self):

        if self.global_tokenizer is None:
            self.global_tokenizer = AutoTokenizer.from_pretrained(self.backbone_diffusion_path, subfolder="tokenizer")

        if self.global_text_encoder is None:
            self.global_text_encoder = CLIPTextModel.from_pretrained(self.backbone_diffusion_path,
                                                                     subfolder="text_encoder").to(device='cuda')

        if self.global_scheduler is None:
            self.global_scheduler = self.one_step_scheduler()

        if self.global_vae is None:
            self.global_vae = AutoencoderKL.from_pretrained(self.backbone_diffusion_path, subfolder="vae")
            self.global_vae = self.skip_connections(self.global_vae)

        if self.global_unet is None:
            self.global_unet = UNet2DConditionModel.from_pretrained(self.backbone_diffusion_path, subfolder="unet")
            p_ckpt = download_url()
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                          target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                                         target_modules=sd["vae_lora_target_modules"])
            self.global_vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = self.global_vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            self.global_vae.load_state_dict(_sd_vae)
            self.global_unet.add_adapter(unet_lora_config)
            _sd_unet = self.global_unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            self.global_unet.load_state_dict(_sd_unet, strict=False)

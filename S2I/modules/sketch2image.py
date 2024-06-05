from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from S2I.modules.models import PrimaryModel
import gc
import torch
import warnings

warnings.filterwarnings("ignore")


class Sketch2Image(PrimaryModel):
    def __init__(self):
        super().__init__()
        self.from_pretrained()
        self.tokenizer = self.global_tokenizer
        self.text_encoder = self.global_text_encoder
        self.scheduler = self.global_scheduler
        self.unet = self.global_unet
        self.vae = self.global_vae
        self.vae.decoder.gamma = 1
        self.timestep = torch.tensor([999], device="cuda").long()

        # Move only necessary parts to GPU
        self.text_encoder.to("cuda")
        self.unet.to("cuda")

    def generate(self, c_t, prompt=None, prompt_tokens=None, r=1.0, noise_map=None, half_model=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        if half_model == 'fp16':
            with torch.autocast:
                if prompt is not None:
                    caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                                    padding="max_length", truncation=True,
                                                    return_tensors="pt").input_ids.cuda()
                    caption_enc = self.text_encoder(caption_tokens)[0]
                else:
                    caption_enc = self.text_encoder(prompt_tokens)[0]

                self.unet.set_adapters(["default"], weights=[r])
                set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])

                # Move VAE temporarily to GPU for encoding
                self.vae.to("cuda")
                encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
                self.vae.to("cpu")  # Move VAE back to CPU to save GPU memory

                unet_input = encoded_control * r + noise_map * (1 - r)
                self.unet.conv_in.r = r
                unet_output = self.unet(unet_input, self.timestep, encoder_hidden_states=caption_enc).sample
                self.unet.conv_in.r = None
                x_denoise = self.scheduler.step(unet_output, self.timestep, unet_input, return_dict=True).prev_sample

                # Move VAE back to GPU for decoding
                self.vae.to("cuda")
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
                self.vae.decoder.gamma = r
                output_image = self.vae.decode(x_denoise / self.vae.config.scaling_factor).sample.clamp(-1, 1)
                self.vae.to("cpu")  # Move VAE back to CPU after decoding
                torch.cuda.empty_cache()
                gc.collect()
                return output_image
        else:
            if prompt is not None:
                caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                                padding="max_length", truncation=True,
                                                return_tensors="pt").input_ids.cuda()
                caption_enc = self.text_encoder(caption_tokens)[0]
            else:
                caption_enc = self.text_encoder(prompt_tokens)[0]

            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])

            # Move VAE temporarily to GPU for encoding
            self.vae.to("cuda")
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            self.vae.to("cpu")  # Move VAE back to CPU to save GPU memory

            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timestep, encoder_hidden_states=caption_enc).sample
            self.unet.conv_in.r = None
            x_denoise = self.scheduler.step(unet_output, self.timestep, unet_input, return_dict=True).prev_sample

            # Move VAE back to GPU for decoding
            self.vae.to("cuda")
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = self.vae.decode(x_denoise / self.vae.config.scaling_factor).sample.clamp(-1, 1)
            self.vae.to("cpu")  # Move VAE back to CPU after decoding
            torch.cuda.empty_cache()
            gc.collect()
            return output_image


# if __name__ == '__main__':
#     Sketch2Image()

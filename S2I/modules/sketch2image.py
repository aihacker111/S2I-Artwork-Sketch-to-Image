from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from S2I.modules.models import PrimaryModel
import gc
import torch
import warnings

warnings.filterwarnings("ignore")


class Sketch2Image(PrimaryModel):
    def __init__(self):
        super().__init__()
        self.timestep = torch.tensor([999], device="cuda").long()
        self.unet, self.vae, self.tokenizer, self.text_encoder, self.scheduler = self.from_pretrained()

    def generate(self, c_t, prompt=None, prompt_tokens=None, r=1.0, noise_map=None, half_model=None, model_name=None):
        self.unet, self.vae = self.initialize_sketch2image(model_name, self.unet, self.vae)
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if half_model == 'fp16':
            return self._generate_fp16(c_t, prompt, prompt_tokens, r, noise_map)
        else:
            return self._generate_full_precision(c_t, prompt, prompt_tokens, r, noise_map)

    def _generate_fp16(self, c_t, prompt, prompt_tokens, r, noise_map):
        with torch.autocast(device_type='cuda'):
            caption_enc = self._get_caption_enc(prompt, prompt_tokens)

            self._set_weights_and_activate_adapters(r)
            self._move_to_gpu(self.vae)
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            self._move_to_cpu(self.vae)

            unet_input = encoded_control * r + noise_map * (1 - r)
            unet_output = self.unet(unet_input, self.timestep, encoder_hidden_states=caption_enc).sample

            x_denoise = self.scheduler.step(unet_output, self.timestep, unet_input, return_dict=True).prev_sample

            self._move_to_gpu(self.vae)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r

            output_image = self.vae.decode(x_denoise / self.vae.config.scaling_factor).sample.clamp(-1, 1)
            self._move_to_cpu(self.vae)
            torch.cuda.empty_cache()
            gc.collect()

            return output_image

    def _generate_full_precision(self, c_t, prompt, prompt_tokens, r, noise_map):
        caption_enc = self._get_caption_enc(prompt, prompt_tokens)

        self._set_weights_and_activate_adapters(r)
        self._move_to_gpu(self.vae)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        self._move_to_cpu(self.vae)

        unet_input = encoded_control * r + noise_map * (1 - r)
        unet_output = self.unet(unet_input, self.timestep, encoder_hidden_states=caption_enc).sample

        x_denoise = self.scheduler.step(unet_output, self.timestep, unet_input, return_dict=True).prev_sample

        self._move_to_gpu(self.vae)
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        self.vae.decoder.gamma = r

        output_image = self.vae.decode(x_denoise / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        self._move_to_cpu(self.vae)
        torch.cuda.empty_cache()
        gc.collect()

        return output_image

    def _get_caption_enc(self, prompt, prompt_tokens):
        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True,
                                            return_tensors="pt").input_ids.cuda()
        else:
            caption_tokens = prompt_tokens.cuda()

        return self.text_encoder(caption_tokens)[0]

    def _set_weights_and_activate_adapters(self, r):
        self.unet.set_adapters(["default"], weights=[r])
        set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])

    @staticmethod
    def _move_to_cpu(module):
        module.to("cpu")

    @staticmethod
    def _move_to_gpu(module):
        module.to("cuda")

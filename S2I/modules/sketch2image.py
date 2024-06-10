from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from S2I.modules.models import PrimaryModel
import gc
import torch
import warnings

warnings.filterwarnings("ignore")


class Sketch2ImagePipeline(PrimaryModel):
    def __init__(self):
        super().__init__()
        self.timestep = torch.tensor([999], device="cuda").long()

    def generate(self, c_t, prompt=None, prompt_tokens=None, r=1.0, noise_map=None, half_model=None, model_name=None):
        self.from_pretrained(model_name=model_name, r=r)
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if half_model == 'fp16':
            return self._generate_fp16(c_t, prompt, prompt_tokens, r, noise_map)
        else:
            return self._generate_full_precision(c_t, prompt, prompt_tokens, r, noise_map)

    def _generate_fp16(self, c_t, prompt, prompt_tokens, r, noise_map):
        with torch.autocast(device_type='cuda'):
            caption_enc = self._get_caption_enc(prompt, prompt_tokens)

            self._set_weights_and_activate_adapters(r)
            self._move_to_gpu(self.global_vae)
            encoded_control = self.global_vae.encode(c_t).latent_dist.sample() * self.global_vae.config.scaling_factor
            self._move_to_cpu(self.global_vae)

            unet_input = encoded_control * r + noise_map * (1 - r)
            unet_output = self.global_unet(unet_input, self.timestep, encoder_hidden_states=caption_enc).sample
            x_denoise = self.global_scheduler.step(unet_output, self.timestep, unet_input, return_dict=True).prev_sample

            self._move_to_gpu(self.global_vae)
            self.global_vae.decoder.incoming_skip_acts = self.global_vae.encoder.current_down_blocks
            self.global_vae.decoder.gamma = r

            output_image = self.global_vae.decode(x_denoise / self.global_vae.config.scaling_factor).sample.clamp(-1, 1)
            self._move_to_cpu(self.global_vae)
            torch.cuda.empty_cache()
            gc.collect()

        return output_image

    def _generate_full_precision(self, c_t, prompt, prompt_tokens, r, noise_map):
        caption_enc = self._get_caption_enc(prompt, prompt_tokens)

        self._set_weights_and_activate_adapters(r)
        self._move_to_gpu(self.global_vae)
        encoded_control = self.global_vae.encode(c_t).latent_dist.sample() * self.global_vae.config.scaling_factor
        self._move_to_cpu(self.global_vae)

        unet_input = encoded_control * r + noise_map * (1 - r)
        unet_output = self.global_unet(unet_input, self.timestep, encoder_hidden_states=caption_enc).sample
        x_denoise = self.global_scheduler.step(unet_output, self.timestep, unet_input, return_dict=True).prev_sample

        self._move_to_gpu(self.global_vae)
        self.global_vae.decoder.incoming_skip_acts = self.global_vae.encoder.current_down_blocks
        self.global_vae.decoder.gamma = r

        output_image = self.global_vae.decode(x_denoise / self.global_vae.config.scaling_factor).sample.clamp(-1, 1)
        self._move_to_cpu(self.global_vae)
        torch.cuda.empty_cache()
        gc.collect()

        return output_image

    def _get_caption_enc(self, prompt, prompt_tokens):
        if prompt is not None:
            caption_tokens = self.global_tokenizer(prompt, max_length=self.global_tokenizer.model_max_length,
                                                   padding="max_length", truncation=True,
                                                   return_tensors="pt").input_ids.cuda()
        else:
            caption_tokens = prompt_tokens.cuda()

        return self.global_text_encoder(caption_tokens)[0]

    def _set_weights_and_activate_adapters(self, r):
        self.global_unet.set_adapters(["default"], weights=[r])
        set_weights_and_activate_adapters(self.global_vae, ["vae_skip"], [r])

    @staticmethod
    def _move_to_cpu(module):
        module.to("cpu")

    @staticmethod
    def _move_to_gpu(module):
        module.to("cuda")

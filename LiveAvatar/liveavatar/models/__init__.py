from .wan.wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper, CausalS2VDiffusionWrapper

from transformers.models.t5.modeling_t5 import T5Block


DIFFUSION_NAME_TO_CLASS = {
    "wan": WanDiffusionWrapper,
    "causal_s2v": CausalS2VDiffusionWrapper
}


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


TEXTENCODER_NAME_TO_CLASS = {
    "wan": WanTextEncoder,
    "causal_wan": WanTextEncoder,
    "causal_s2v": WanTextEncoder
}


def get_text_encoder_wrapper(model_name):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


VAE_NAME_TO_CLASS = {
    "wan": WanVAEWrapper,
    "causal_wan": WanVAEWrapper,   # TODO: Change the VAE to the causal version
    "causal_s2v": WanVAEWrapper
}


def get_vae_wrapper(model_name):
    return VAE_NAME_TO_CLASS[model_name]




BLOCK_NAME_TO_BLOCK_CLASS = {
    "T5Block": T5Block
}


def get_block_class(model_name):
    return BLOCK_NAME_TO_BLOCK_CLASS[model_name]

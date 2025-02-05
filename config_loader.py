from pydantic import BaseModel, Field
from typing import List, Union, Literal
from pathlib import Path
import yaml
import json


class TrainingConfig(BaseModel):
    """
    Training configuration parameters.
    """

    out_dir: str = Field(..., description="Directory for output files.")
    log_interval: int = Field(..., description="Interval (in steps) for logging.")
    save_interval: int = Field(
        ..., description="Interval (in steps) for saving checkpoints."
    )
    val_interval: int = Field(..., description="Interval (in steps) for validation.")
    save_epoch_interval: int = Field(
        ..., description="Interval (in epochs) for saving checkpoints."
    )
    device: str = Field(..., description="Computational device (e.g., 'cuda').")
    mixed_precision: str = Field(..., description="accelerator use bf16 or fp16 or no")


class TrainingPlanConfig(BaseModel):
    """
    Training plan configuration that defines the number of epochs for different stages.
    """

    first: int = Field(
        ..., description="Number of epochs for the first training stage."
    )
    first_tma: int = Field(
        ..., description="Number of epochs for the first stage with TMA."
    )
    second: int = Field(
        ..., description="Number of epochs for the second training stage."
    )
    second_style: int = Field(
        ..., description="Number of epochs for the second stage with style training."
    )
    second_joint: int = Field(
        ..., description="Number of epochs for the second stage with joint training."
    )


class DatasetConfig(BaseModel):
    """
    Dataset configuration parameters.
    """

    train_data: str = Field(..., description="Path to the training data list.")
    val_data: str = Field(..., description="Path to the validation data list.")
    wav_path: str = Field(..., description="Directory containing WAV files.")
    OOD_data: str = Field(..., description="Path to out-of-domain texts file.")
    min_length: int = Field(
        ..., description="Minimum text length for sampling (used for OOD texts)."
    )


class PreprocessConfig(BaseModel):
    """
    Preprocessing configuration parameters.
    """

    sample_rate: int = Field(..., description="Sample rate for audio preprocessing.")
    n_fft: int = Field(..., description="FFT size for spectrogram computation.")
    win_length: int = Field(
        ..., description="Window length for spectrogram computation."
    )
    hop_length: int = Field(..., description="Hop length for spectrogram computation.")


class ModelConfig(BaseModel):
    """
    General model configuration parameters.
    """

    multispeaker: bool = Field(
        ..., description="Indicates if the model supports multispeaker input."
    )
    n_mels: int = Field(..., description="Number of mel frequency bins.")
    style_dim: int = Field(..., description="Dimension of the style vector.")


class PretrainedConfig(BaseModel):
    """
    Configuration for pretrained models.
    """

    F0_path: str = Field(..., description="Path to the F0 model file.")
    ASR_config: str = Field(..., description="Path to the ASR configuration file.")
    ASR_path: str = Field(..., description="Path to the ASR model file.")
    PLBERT_config: str = Field(
        ..., description="Path to the PLBERT configuration file."
    )
    PLBERT_path: str = Field(..., description="Path to the PLBERT model file.")


class HiFiGANDecoderConfig(BaseModel):
    """
    Configuration for HiFiGAN decoder.
    """

    type: Literal["hifigan"] = "hifigan"
    hidden_dim: int = Field(..., description="Hidden dimension for HiFiGAN.")
    resblock_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for residual blocks."
    )
    upsample_rates: List[int] = Field(
        ..., description="Upsample rates for each upsampling layer."
    )
    upsample_initial_channel: int = Field(
        ..., description="Initial channel count for upsampling."
    )
    resblock_dilation_sizes: List[List[int]] = Field(
        ..., description="Dilation sizes for residual blocks."
    )
    upsample_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for the upsampling layers."
    )


class ISTFTNetDecoderConfig(BaseModel):
    """
    Configuration for iSTFTNet decoder.
    """

    type: Literal["istftnet"] = "istftnet"
    hidden_dim: int = Field(..., description="Hidden dimension for iSTFTNet.")
    resblock_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for residual blocks."
    )
    upsample_rates: List[int] = Field(
        ..., description="Upsample rates for each upsampling layer."
    )
    upsample_initial_channel: int = Field(
        ..., description="Initial channel count for upsampling."
    )
    resblock_dilation_sizes: List[List[int]] = Field(
        ..., description="Dilation sizes for residual blocks."
    )
    upsample_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for the upsampling layers."
    )
    gen_istft_n_fft: int = Field(..., description="FFT size for iSTFT generator.")
    gen_istft_hop_size: int = Field(..., description="Hop size for iSTFT generator.")


class RingformerDecoderConfig(BaseModel):
    """
    Configuration for Ringformer decoder.
    """

    type: Literal["ringformer"] = "ringformer"
    hidden_dim: int = Field(..., description="Hidden dimension for Ringformer.")
    resblock_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for residual blocks."
    )
    upsample_rates: List[int] = Field(
        ..., description="Upsample rates for each upsampling layer."
    )
    upsample_initial_channel: int = Field(
        ..., description="Initial channel count for upsampling."
    )
    resblock_dilation_sizes: List[List[int]] = Field(
        ..., description="Dilation sizes for residual blocks."
    )
    upsample_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for the upsampling layers."
    )
    gen_istft_n_fft: int = Field(..., description="FFT size for iSTFT generator.")
    gen_istft_hop_size: int = Field(..., description="Hop size for iSTFT generator.")


class VocosDecoderConfig(BaseModel):
    """
    Configuration for Vocos decoder.
    """

    type: Literal["vocos"] = "vocos"
    hidden_dim: int = Field(..., description="Hidden dimension for Vocos.")
    intermediate_dim: int = Field(
        ..., description="Intermediate dimension size for Vocos."
    )
    num_layers: int = Field(..., description="Number of layers in Vocos model.")
    gen_istft_n_fft: int = Field(..., description="FFT size for iSTFT generator.")
    gen_istft_hop_size: int = Field(..., description="Hop size for iSTFT generator.")


class TextEncoderConfig(BaseModel):
    """
    Text encoder configuration parameters.
    """

    hidden_dim: int = Field(..., description="Hidden dimension of the text encoder.")
    kernel_size: int = Field(
        ..., description="Kernel size for convolution in the text encoder."
    )
    n_layer: int = Field(..., description="Number of layers in the text encoder.")
    n_token: int = Field(..., description="Number of phoneme tokens.")


class EmbeddingEncoderConfig(BaseModel):
    """
    Embedding encoder configuration parameters.
    This encoder (which may also act as a prosody encoder) generates a style embedding from audio.
    """

    dim_in: int = Field(..., description="Input dimension for the embedding encoder.")
    hidden_dim: int = Field(
        ..., description="Hidden dimension for the embedding encoder."
    )
    skip_downsamples: bool = Field(
        ..., description="Flag indicating whether to skip downsampling."
    )


class ProsodyPredictorConfig(BaseModel):
    """
    Prosody predictor configuration parameters.
    """

    hidden_dim: int = Field(
        ..., description="Hidden dimension for the prosody predictor."
    )
    n_layer: int = Field(..., description="Number of layers in the prosody predictor.")
    max_dur: int = Field(..., description="Maximum duration of a single phoneme.")
    dropout: float = Field(..., description="Dropout rate for the prosody predictor.")


class SlmConfig(BaseModel):
    """
    Speech Language Model (SLM) configuration parameters.
    """

    model: str = Field(..., description="Identifier or path for the SLM model.")
    sr: int = Field(..., description="Sampling rate used by the SLM.")
    hidden: int = Field(..., description="Hidden dimension of the SLM.")
    nlayers: int = Field(..., description="Number of layers in the SLM.")
    initial_channel: int = Field(
        ..., description="Initial channel count for the SLM discriminator head."
    )


class DiffusionTransformerConfig(BaseModel):
    """
    Transformer configuration parameters for the diffusion model.
    """

    num_layers: int = Field(..., description="Number of transformer layers.")
    num_heads: int = Field(
        ..., description="Number of attention heads in each transformer layer."
    )
    head_features: int = Field(
        ..., description="Feature dimension for each attention head."
    )
    multiplier: int = Field(
        ..., description="Multiplier for scaling transformer outputs."
    )


class DiffusionDistConfig(BaseModel):
    """
    Distribution configuration for the diffusion model.
    """

    sigma_data: float = Field(
        ...,
        description="Sigma value for the diffusion distribution (if not estimated).",
    )
    estimate_sigma_data: bool = Field(
        ..., description="If True, sigma_data is estimated from the current batch."
    )
    mean: float = Field(..., description="Mean of the diffusion distribution.")
    std: float = Field(
        ..., description="Standard deviation of the diffusion distribution."
    )


class DiffusionConfig(BaseModel):
    """
    Style diffusion model configuration parameters.
    """

    embedding_mask_proba: float = Field(
        ..., description="Probability of masking embeddings in the diffusion process."
    )
    transformer: DiffusionTransformerConfig = Field(
        ..., description="Transformer configuration for diffusion."
    )
    dist: DiffusionDistConfig = Field(
        ..., description="Diffusion distribution configuration."
    )


class LossWeightConfig(BaseModel):
    """
    Loss weight configuration for various loss components.
    """

    mel: float = Field(..., description="Weight for mel reconstruction loss.")
    gen: float = Field(..., description="Weight for generator loss.")
    slm: float = Field(..., description="Weight for SLM feature matching loss.")
    mono: float = Field(..., description="Weight for monotonic alignment loss.")
    s2s: float = Field(..., description="Weight for sequence-to-sequence loss.")
    F0: float = Field(..., description="Weight for F0 reconstruction loss.")
    norm: float = Field(..., description="Weight for norm reconstruction loss.")
    duration: float = Field(..., description="Weight for duration loss.")
    duration_ce: float = Field(
        ..., description="Weight for duration cross-entropy loss."
    )
    style: float = Field(..., description="Weight for style reconstruction loss.")
    diffusion: float = Field(
        ..., description="Weight for score matching (diffusion) loss."
    )


class OptimizerConfig(BaseModel):
    """
    Optimizer configuration parameters.
    """

    lr: float = Field(..., description="General learning rate.")
    bert_lr: float = Field(..., description="Learning rate for the PLBERT model.")
    ft_lr: float = Field(..., description="Learning rate for acoustic modules.")


class SlmAdvConfig(BaseModel):
    """
    SLM adversarial training configuration parameters.
    """

    min_len: int = Field(..., description="Minimum length of samples.")
    max_len: int = Field(..., description="Maximum length of samples.")
    batch_percentage: float = Field(
        ...,
        description="Fraction of the original batch size to prevent out-of-memory errors.",
    )
    iter: int = Field(
        ...,
        description="Number of generator update iterations before a discriminator update.",
    )
    thresh: float = Field(
        ..., description="Gradient norm threshold above which scaling is applied."
    )
    scale: float = Field(
        ..., description="Scaling factor for gradients from SLM discriminators."
    )
    sig: float = Field(..., description="Sigma for differentiable duration modeling.")


class Config(BaseModel):
    """
    Top-level configuration model that encompasses all settings.
    """

    training: TrainingConfig = Field(
        ..., description="Training configuration parameters."
    )
    training_plan: TrainingPlanConfig = Field(
        ..., description="Training plan configuration parameters."
    )
    dataset: DatasetConfig = Field(..., description="Dataset configuration parameters.")
    preprocess: PreprocessConfig = Field(
        ..., description="Preprocessing configuration parameters."
    )
    model: ModelConfig = Field(
        ..., description="General model configuration parameters."
    )
    pretrained: PretrainedConfig = Field(
        ..., description="Pretrained model configuration parameters."
    )
    decoder: Union[
        HiFiGANDecoderConfig,
        ISTFTNetDecoderConfig,
        RingformerDecoderConfig,
        VocosDecoderConfig,
    ] = Field(..., description="Decoder (vocoder) configuration parameters.")
    text_encoder: TextEncoderConfig = Field(
        ..., description="Text encoder configuration parameters."
    )
    embedding_encoder: EmbeddingEncoderConfig = Field(
        ..., description="Embedding encoder configuration parameters."
    )
    prosody_predictor: ProsodyPredictorConfig = Field(
        ..., description="Prosody predictor configuration parameters."
    )
    slm: SlmConfig = Field(
        ..., description="Speech Language Model (SLM) configuration parameters."
    )
    diffusion: DiffusionConfig = Field(
        ..., description="Style diffusion model configuration parameters."
    )
    loss_weight: LossWeightConfig = Field(
        ..., description="Loss weight configuration for various loss components."
    )
    optimizer: OptimizerConfig = Field(
        ..., description="Optimizer configuration parameters."
    )
    slmadv_params: SlmAdvConfig = Field(
        ..., description="SLM adversarial training configuration parameters."
    )


class TrainContext:
    def __init__(self):
        config: Config = None
        pass


def load_config_yaml(config_path: str) -> Config:
    """
    Load a configuration file from the specified path.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    path = Path(config_path)
    # Load the YAML file into a dictionary
    with path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    # Parse and validate the configuration dictionary
    return Config.model_validate(config_dict)


def dump_to_string(config: Config) -> str:
    """
    Dump the configuration object to a string.

    Args:
        config (Config): Configuration object to dump.

    Returns:
        str: JSON string representation of the configuration object.
    """
    return config.model_dump_json()  # Dumps as a compact single-line JSON string


def load_from_string(config_str: str) -> Config:
    """
    Load a configuration object from a JSON string.

    Args:
        config_str (str): JSON string representation of the configuration object.

    Returns:
        Config: Parsed configuration object.
    """
    # Convert JSON string to a dictionary and validate it with the Config model
    return Config.model_validate(json.loads(config_str))


# Example usage:
if __name__ == "__main__":
    # Path to the YAML configuration file
    config_path = Path("Configs/new.config.yml")

    # Load the YAML file into a dictionary
    with config_path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
    # Parse and validate the configuration dictionary
    config = Config.model_validate(config_dict)

    # For Pydantic v2, use model_dump_json (or use json.dumps(config.model_dump(), indent=2))
    test = dump_to_string(config)
    print(test)

    # Load a configuration object from a JSON string
    config_loaded = load_from_string(test)
    print(config_loaded)

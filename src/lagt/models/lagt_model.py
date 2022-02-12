from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model

from lagt.models.mha import MultiHeadAttention
from lagt.models.video_quality_transformer import VideoQualityTransformer


def create_model(clip_length=16, feature_length=1280, transformer_params=(2, 64, 4, 64), dropout_rate=0.1):
    """
    Create the LAGT-PHIQNet model for NR-VQA
    :param clip_length: clip length
    :param feature_length: length of frame PHIQNet features, default is 1280=5*256
    :param transformer_params: Transformer parameters
    :param dropout_rate: dropout rate for both 1D CNN and Transformer
    :return: the LAGT-PHIQNet model
    """
    local_mha = MultiHeadAttention(d_model=256, num_heads=4)

    input_shape = (None, clip_length, feature_length)

    inputs = Input(shape=input_shape)

    x = TimeDistributed(local_mha)(inputs)

    transformer = VideoQualityTransformer(
        num_layers=transformer_params[0],
        d_model=transformer_params[1],
        num_heads=transformer_params[2],
        mlp_dim=transformer_params[3],
        dropout=dropout_rate,
    )
    x = transformer(x)

    model = Model(inputs=inputs, outputs=x)

    return model

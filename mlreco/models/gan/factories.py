def gan_model_dict():
    """
    Imports and returns dictionary of valid gan sub models.

    Args:
    Returns:
        dict: Dictionary of valid gan sub models
    """
    from . import gan_uresnet_generator
    from . import gan_encoder_discriminator

    sub_models = {
        'uresnet': gan_uresnet_generator.GANUResNetGenerator,
        'encoder': gan_encoder_discriminator.GANEncoderDiscriminator,
    }
    return sub_models


def gan_construct(cfg):
    """
    Constructor of GAN sub models
    """
    sub_models = gan_model_dict()
    name = cfg['name']
    if not name in sub_models:
        raise Exception("Unknown GAN sub-model name provided:", name)

    return sub_models[name](cfg)

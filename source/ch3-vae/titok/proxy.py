from taming.models.vqgan import VQModel

def load_vqgan_model(config_path, checkpoint_path):
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    vqgan = VQModel(**config.model.params)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    vqgan.load_state_dict(state_dict)
    return vqgan

vqgan_model = load_vqgan_model('vqgan_imagenet_f16_1024.yaml', 'vqgan_imagenet_f16_1024.ckpt').to(device)

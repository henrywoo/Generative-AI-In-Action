import torch
from pprint import pprint
from dataset import create_dataset
from model.UNet import UNet
from sampler.gaussian_trainer import GaussianDiffusionTrainer
from sampler.tools import train_one_epoch, load_yaml
from sampler.callbacks import ModelCheckpoint
from hiq import print_model
from hiq.cv_torch import get_cv_dataset


def get_dl(name='cifar10'):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    loader_params = dict(
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    return get_cv_dataset(path=name,
                          image_size=32,
                          split='train',
                          batch_size=2,
                          num_workers=2,
                          transform=transform,
                          return_type="pair",
                          return_loader=True,
                          **loader_params
                          )


def train(config):
    resume = config["resume"]
    if resume:
        cp = torch.load(config["resume_path"])
        #cp['config']["Dataset"]["batch_size"] = config["Dataset"]["batch_size"]
        config = cp["config"]
    pprint(config)
    device = torch.device(config["device"])
    loader = get_dl()  # create_dataset(**config["Dataset"])
    start_epoch = 1
    model = UNet(**config["Model"]).to(device)
    print_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)
    model_checkpoint = ModelCheckpoint(**config["Callback"])
    if resume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1
    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch)
        model_checkpoint.step(loss, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_checkpoint.state_dict())


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)

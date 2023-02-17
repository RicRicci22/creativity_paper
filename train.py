import argparse
import collections
from parse_config import ConfigParser
import random

import models.creativity_model as module_arch
import data_loader.data_loaders as module_data
import models.loss as module_loss
import models.metric as module_metric
import torch
import numpy as np
from trainer import Trainer
from utils import prepare_device

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj(
        "data_loader", module_data, training=True, overfitting=False, generator=g
    )
    valid_data_loader = data_loader.split_validation()

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])

    vocab_size = data_loader.vocab_size
    # print(vocab_size)
    model = config.init_obj(
        "arch",
        module_arch,
        vocab_size=vocab_size,
        sos_token=2,
        eos_token=3,
        device=device,
    )
    logger.info(model)

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = getattr(module_loss, config["loss"])

    metrics = [getattr(module_metric, met) for met in config["metrics"]]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)

    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Creativity Model")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

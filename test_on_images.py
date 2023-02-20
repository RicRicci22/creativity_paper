import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import models.loss as module_loss
import models.metric as module_metric
import models.creativity_model as module_arch
from parse_config import ConfigParser
from torch.nn.utils.rnn import pack_padded_sequence
import os
from torchvision.io import read_image, ImageReadMode
import streamlit as st
from torchvision.transforms import ToPILImage


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        image_dir=config["data_loader"]["args"]["image_dir"],
        questions_path=config["data_loader"]["args"]["questions_path"],
        test_imgs=config["data_loader"]["args"]["test_imgs"],
        batch_size=16,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=1,
    )

    # build model architecture
    vocab_size = data_loader.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.init_obj(
        "arch",
        module_arch,
        vocab_size=vocab_size,
        sos_token=2,
        eos_token=3,
        device=device,
    )

    model = model.to(device)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare model for testing
    model.eval()
    topil = ToPILImage()
    with torch.no_grad():
        for image in data_loader.dataset.image_names:
            image_path = os.path.join(data_loader.dataset.image_dir, image)
            image = read_image(image_path, mode=ImageReadMode.RGB) / 255.0
            st.image(topil(image))
            image = image.unsqueeze(0)  # to simulate batch size
            image = image.to(device)
            question = model.sample(image, max_len=50, multinomial=False)
            # question = model.beam_decode(image, max_len=50, beam_width=3, topk=1)
            # for b in question:
            #     for topk in b:
            #         st.text(topk[0])
            #         st.text(data_loader.tokenizer.decode(topk[1]))
            for i, question in enumerate(question):
                # Convert to list
                question = question.tolist()
                eos_position = question.index(3)
                if eos_position != -1:
                    question = question[:eos_position]
                    st.text(data_loader.tokenizer.decode(question))
                else:
                    st.text(data_loader.tokenizer.decode(question))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
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

    config = ConfigParser.from_args(args)
    main(config)

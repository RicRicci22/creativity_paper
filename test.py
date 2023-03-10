import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import models.loss as module_loss
import models.metric as module_metric
import models.creativity_model as module_arch
from parse_config import ConfigParser
from torch.nn.utils.rnn import pack_padded_sequence


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
        eos_token=8,
        device=device,
    )
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target, lenghts) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            if config["arch"]["type"] == "Im2QModel":
                output = model(data, target, lenghts)
            else:
                output, mean, logvar = model(data, target, lenghts)

            target = pack_padded_sequence(
                target[:, 1:], [l - 1 for l in lenghts], batch_first=True
            )[0]

            #
            # save sample images, or do something with output here
            # Examples ####################################################
            print("Examples from training:")
            sampled_questions = model.beam_decode(
                data, max_len=50, beam_width=3, topk=20
            )

            for i, question in enumerate(sampled_questions):
                # Convert to list
                for q in question:
                    print(q[0])
                    print(data_loader.tokenizer.decode(q[1]))
                break
            exit()
            # question = question.tolist()
            # eos_position = question.index(3)
            # print("Sampled question: ", str(i))
            # if eos_position != -1:
            #     question = question[:eos_position]
            #     print(data_loader.tokenizer.decode(question))
            # else:
            #     print(data_loader.tokenizer.decode(question))
            ###############################################################
            #

            # computing loss, metrics on test set
            rec_loss, _ = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += rec_loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    logger.info(log)


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

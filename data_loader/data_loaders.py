from torchvision import transforms
from base.base_dataloader import BaseDataLoader
from dataset.datasets import UAVDataset
import torch
from utils import UAVCollator
from transformers import BertTokenizer


class UAVDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, image_dir, questions_path, test_imgs, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.collate_fn = UAVCollator(self.tokenizer)
        self.dataset = UAVDataset(image_dir, questions_path, test_imgs, transform=trsfm, training=training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)
    
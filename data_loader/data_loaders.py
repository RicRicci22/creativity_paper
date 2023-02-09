from torchvision import transforms
from base.base_dataloader import BaseDataLoader
from dataset.datasets import UAVDataset
import torch


class UAVDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, image_dir, questions_path, test_imgs, collate_fn, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.collate_fn = collate_fn
        self.dataset = UAVDataset(image_dir, questions_path, test_imgs, transform=trsfm, training=training)
        
        # Create collate function
        def collate_fn(batch):
            images, questions = zip(*batch)
            images = torch.stack(images)
            questions = self.tokenizer(questions, padding=True, return_tensors="pt")
            return images, questions

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)
    
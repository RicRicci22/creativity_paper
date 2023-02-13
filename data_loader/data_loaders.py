from torchvision import transforms
from base.base_dataloader import BaseDataLoader
from dataset.datasets import UAVDataset
from utils import UAVCollator
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


class UAVDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, image_dir, questions_path, test_imgs, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print('Building tokenizer..')
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True) # Picked just to test
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"])
        tokenizer.train(files=["dataset/questions.txt"], trainer=trainer)
        sos_token_id = tokenizer.token_to_id("[SOS]")
        eos_token_id = tokenizer.token_to_id("[EOS]")
        tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[SOS] $A [EOS]",
        special_tokens=[("[SOS]", sos_token_id), ("[EOS]", eos_token_id)],
        )
        print("Done! Tokenizer built!")

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.collate_fn = UAVCollator(self.tokenizer)
        self.dataset = UAVDataset(image_dir, questions_path, test_imgs, transform=trsfm, training=training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)
    
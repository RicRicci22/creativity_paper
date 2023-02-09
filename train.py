from models.creativity_model import CreativityModel, CreativityEncoder
from data_loader.data_loaders import UAVDataLoader
from transformers import BertTokenizer
from utils import UAVCollator
import torch

if __name__=="__main__":
    # # Test the dataloader
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # collator = UAVCollator(tokenizer=tokenizer)
    # data_loader = UAVDataLoader(image_dir="dataset/images", questions_path="dataset/UAV_summaries_and_questions_formatted.pkl", test_imgs="dataset/filenames_test.txt", collate_fn = collator, batch_size=8)
    # for i, (images, questions) in enumerate(data_loader):
    #     print(images.shape)
    #     print(questions['input_ids'].shape)
    #     break

    # Test the creativity encoder
    model = CreativityModel(backbone_name="resnet18", hidden_size=512, latent_size=20, vocab_size=30522, sos_token=10)
    dummy_batch = torch.randn((8, 3, 224, 224))
    dummy_questions = torch.randint(0, 30522, (8, 20))
    # Forward pass of the model
    out = model(dummy_batch, dummy_questions)
    print(out.shape)

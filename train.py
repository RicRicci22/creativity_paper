from models.creativity_model import CreativityModel
from data_loader.data_loaders import UAVDataLoader
from transformers import BertTokenizer
from utils import UAVCollator

if __name__=="__main__":
    # Test the dataloader
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collator = UAVCollator(tokenizer=tokenizer)
    data_loader = UAVDataLoader(image_dir="dataset/images", questions_path="dataset/UAV_summaries_and_questions_formatted.pkl", test_imgs="dataset/filenames_test.txt", collate_fn = collator, batch_size=8)
    for i, (images, questions) in enumerate(data_loader):
        print(images.shape)
        print(questions)
        break
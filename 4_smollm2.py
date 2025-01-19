import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load environment variables from .env file
load_dotenv()

class StreamingDataLoader:
    def __init__(self, batch_size=16, tokenizer_name='smollm-encoding'):
        huggingface_token = os.getenv('HUGGINGFACE_TOKEN')  # Access the Hugging Face token
        self.dataset = load_dataset(path='HuggingFaceTB/smollm-corpus',
                                    name='cosmopedia-v2',
                                    split='train',
                                    streaming=True,
                                    token=huggingface_token)  # Pass the token while loading the dataset
        self.batch_size = batch_size
        self.iterator = iter(self.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

    def next_batch(self):
        batch = []
        try:
            for _ in range(self.batch_size):
                text = next(self.iterator)['text']
                encoded_text = self.tokenizer(text, return_tensors='pt').to("mps")  # Tokenize the text
                batch.append(encoded_text['input_ids'][0])
        except StopIteration:
            self.iterator = iter(self.dataset)  # Reset iterator if end is reached
            batch = self.next_batch()  # Get the next batch again
        return batch

# Example usage
data_loader = StreamingDataLoader()

for batch in data_loader.next_batch():
    # Process your batch here
    print(batch)  # Replace with your processing logic
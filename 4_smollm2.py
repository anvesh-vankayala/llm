import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer
# Load environment variables from .env file
load_dotenv()



class GPTConfig:
    block_size: int = 2048 # max sequence length
    vocab_size: int = 49152 # number of tokens: ~50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 9 # number of layers
    n_head: int = 9 # attention heads
    n_embd: int = 768 # embedding dimension

class StreamingDataLoader:
    def __init__(self, batch_size=16, token_length=2048):
        huggingface_token = os.getenv('HUGGINGFACE_TOKEN')  # Access the Hugging Face token
        self.dataset = load_dataset(path='HuggingFaceTB/smollm-corpus',
                                    name='cosmopedia-v2',
                                    split='train',
                                    streaming=True,
                                    token=huggingface_token)  # Pass the token while loading the dataset
        self.batch_size = batch_size
        self.token_length = token_length
        self.iterator = iter(self.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        self.current_position = 0

    def next_batch(self):
        B, T = self.batch_size, self.token_length
        batch = []
        
        for _ in range(55): ## as this is streaming data, avg 20 iterations to get a batch more than 16 X 2048 tokens.
            text = next(self.iterator)['text']
            encoded_text = self.tokenizer(text, return_tensors='pt').to("mps")  # Tokenize the text
            batch.append(encoded_text['input_ids'][0])
            
        batch_merged = torch.cat(batch, dim=0)  # Unpack and merge the lists to a new tensor across dim=0
        buf = batch_merged[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position = 0
        return x, y
# Example usage
data_loader = StreamingDataLoader()

for batch in data_loader.next_batch():
    # Process your batch here
    print(batch.shape)  # Replace with your processing logic
    break

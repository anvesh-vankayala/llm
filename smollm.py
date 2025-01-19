import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import hf_hub_download

class ModelHandler:
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-135M"):
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

    def generate_text(self, prompt="Gravity is"):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("mps")
        outputs = self.model.generate(inputs)
        return self.tokenizer.decode(outputs[0])

# Example usage
model_handler = ModelHandler()
generated_text = model_handler.generate_text()
print(generated_text)

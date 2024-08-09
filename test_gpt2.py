from train_gpt2 import GPT, GPTConfig
import torch

# Load the model configuration
config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
device = 'cuda'

# Load the saved model
model_path = 'model/gpt2_124M_bfloat16.pt'
tokenizer_path = 'model/gpt2_tokenizer.pt'
debug_state_path = 'model/gpt2_124M_debug_state.pt'

model = GPT(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

enc = torch.load(tokenizer_path)
debug_state = torch.load(debug_state_path)
x, y, logits, loss, grads = debug_state['x'], debug_state['y'], debug_state['logits'], debug_state['loss'], debug_state['grads']

# Generate text
start_text = "The White man worked as a captain of a ship"
input_ids = torch.tensor(enc.encode(start_text), dtype=torch.long, device=device)[None, ...]
output_ids = model.generate(input_ids, max_new_tokens=100, temperature=1.0, top_k=40)
generated_text = enc.decode(output_ids[0].tolist())
print(generated_text)

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
#
# enc = AutoTokenizer.from_pretrained("")
# model = AutoModelForCausalLM.from_pretrained("")
# device = 'cuda'
# model.to(device)
#
# # Generate text
# start_text = "Hello, I'm a language model."
# input_ids = torch.tensor(enc.encode(start_text), dtype=torch.long, device=device)[None, ...]
# output_ids = model.generate(input_ids, max_new_tokens=100, temperature=1.0, top_k=40)
# generated_text = enc.decode(output_ids[0].tolist())
# print(generated_text)
"""Test single model load to check GPU memory usage."""
import sys
import os
import torch

sys.path.append('scripts')

print("="*60)
print("GPU Memory Test - Before Loading")
print("="*60)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
else:
    print("CUDA not available")
    sys.exit(1)

print("\n" + "="*60)
print("Loading Model...")
print("="*60)

from utils import get_model

# Load with 8-bit quantization to save memory
model, repo_dir = get_model(load_in_8bit=True)

print("\n" + "="*60)
print("GPU Memory Test - After Loading")
print("="*60)
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")

print("\n" + "="*60)
print("Testing Single Inference")
print("="*60)

test_image = "http://images.cocodataset.org/val2017/000000039769.jpg"
prompt = "What animals are in this image?"

result = model.inference(prompt, test_image, task="general", enable_thinking=False)

print(f"\nResult: {result['answer'][:100]}...")

print("\n" + "="*60)
print("GPU Memory Test - After Inference")
print("="*60)
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")

print("\n✓ Test completed successfully - No OOM!")

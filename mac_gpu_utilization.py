import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use the GPU with MPS
    print("Using MPS GPU")
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(f"Total GPU Memory: {total_memory / (1024 ** 2):.2f} MB")
    
    # Check GPU memory usage
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Memory Allocated: {allocated_memory / (1024 ** 2):.2f} MB")
    
    # Calculate and print free memory
    free_memory = total_memory - allocated_memory
    print(f"Memory Free: {free_memory / (1024 ** 2):.2f} MB")
    
    # Calculate and print GPU utilization
    utilization = torch.cuda.memory_stats(device)['allocated_bytes.all.peak'] / total_memory * 100
    print(f"GPU Utilization: {utilization:.2f}%")
else:
    print("No MPS GPU available, using CPU.")
    total_memory = 0  # Set total memory to 0 for CPU
    allocated_memory = 0  # Set allocated memory to 0 for CPU
    free_memory = total_memory - allocated_memory  # Free memory is also 0
    print(f"Memory Free: {free_memory:.2f} MB")
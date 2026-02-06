import torch

def main():
    print("=== PyTorch CUDA Check ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (torch):", torch.version.cuda)
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
    print("cuDNN version:", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        print("\n=== GPU Info ===")
        n = torch.cuda.device_count()
        print("Device count:", n)
        for i in range(n):
            print(f"[{i}] Name:", torch.cuda.get_device_name(i))
            print("    Capability:", torch.cuda.get_device_capability(i))
            print("    Total memory (GB):", round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2))

        # Test tạo tensor trên GPU và chạy phép toán
        device = torch.device("cuda:0")
        x = torch.randn(2000, 2000, device=device)
        y = torch.randn(2000, 2000, device=device)
        z = x @ y  # matmul
        torch.cuda.synchronize()
        print("\nTensor test OK. z device:", z.device)
    else:
        print("\nCUDA không khả dụng. PyTorch đang chạy CPU.")

if __name__ == "__main__":
    main()

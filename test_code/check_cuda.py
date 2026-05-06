import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    x = torch.tensor([1.0]).cuda()
    print('Tensor on GPU:', x)
else:
    print('No CUDA - check driver / torch build')

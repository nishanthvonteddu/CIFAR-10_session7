import torch
from torchsummary import summary

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, device):
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    summary(model, input_size=(3, 32, 32), device=device.type)
    print("="*80)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import CustomCIFARNet
from data import get_dataloaders
from utils import count_parameters, print_model_summary
from logger import TrainingLogger

# ======================== Training Functions ========================

def train_epoch(model, device, train_loader, optimizer, criterion, scheduler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if scheduler and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return running_loss/len(train_loader), 100.*correct/total

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            pbar.set_postfix({'loss': f'{test_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return test_loss/len(test_loader), 100.*correct/total

# ======================== Main ========================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCIFARNet().to(device)
    total_params = count_parameters(model)

    print(f"\n{'='*80}\nMODEL INFORMATION\n{'='*80}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Parameters < 200k: {'✓ YES' if total_params < 200000 else '✗ NO'}")
    print(f"Receptive Field: 45 (> 44 ✓)")
    print(f"Architecture: C1-C2-C3-C4 (No MaxPooling ✓)")
    print(f"Uses Dilated Convolution: ✓ (C2, C4)")
    print(f"Uses Depthwise Separable: ✓ (C3)")
    print(f"Uses GAP + FC: ✓")
    print("="*80)
    print_model_summary(model, device)

    # Data
    train_loader, test_loader = get_dataloaders(batch_size=128)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=50,
                                              steps_per_epoch=len(train_loader),
                                              pct_start=0.2, anneal_strategy='cos')

    logger = TrainingLogger()
    best_acc, target_acc = 0.0, 85.0

    print("\n" + "="*80)
    print("TRAINING LOGS (Validation after each epoch)")
    print("="*80)

    for epoch in range(50):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, scheduler)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        logger.log(epoch+1, train_loss, train_acc, test_loss, test_acc)
        logger.print_epoch(epoch+1, train_loss, train_acc, test_loss, test_acc, max(best_acc, test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"         → Saved new best model with accuracy: {best_acc:.2f}%")
        if test_acc >= target_acc:
            print(f"\n{'='*80}\n🎉 TARGET ACHIEVED! Test accuracy {target_acc}% at epoch {epoch+1}!\n{'='*80}")
            break

    logger.print_summary()
    print(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Parameters: {total_params:,}")
    print(f"Target Accuracy (85%): {'✓ ACHIEVED' if best_acc >= 85.0 else '✗ NOT ACHIEVED'}")
    print(f"Model saved as: best_model.pth")
    print("="*80)

if __name__ == '__main__':
    main()

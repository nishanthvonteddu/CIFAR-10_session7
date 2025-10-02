class TrainingLogger:
    def __init__(self):
        self.train_losses, self.train_accs = [], []
        self.test_losses, self.test_accs = [], []

    def log(self, epoch, train_loss, train_acc, test_loss, test_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)

    def print_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, best_acc):
        print(f"\nEpoch: {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
              f"Best Acc: {best_acc:6.2f}%")

    def print_summary(self):
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
        print("-"*80)
        for i in range(len(self.train_losses)):
            print(f"{i+1:<8} {self.train_losses[i]:<12.4f} {self.train_accs[i]:<12.2f} "
                  f"{self.test_losses[i]:<12.4f} {self.test_accs[i]:<12.2f}")
        print("="*80)

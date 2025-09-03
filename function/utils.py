import torch
import torch.nn as nn

class COSINANNEALINGWARMRESTARTS():
    def __init__(self, optimizer, T_0=10, T_mult=1, eta_min=1e-5):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            # patience=10,
            T_0= 10, 
            eta_min=1e-6,
            T_mult=2,
            # verbose=True
        )

    def __call__(self, val_loss):
        self.scheduler.step(val_loss)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)

        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"==>>> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                print("==>>> Early stopping triggered!")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"==>>> Validation loss decreased ({self.best_loss:.6f} → {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)

### Val_Acc
from collections import defaultdict

def compute_classwise_accuracy(preds, labels, num_classes):
    correct = defaultdict(int)
    total = defaultdict(int)

    for pred, label in zip(preds, labels):
        total[label] += 1
        if pred == label:
            correct[label] += 1

    acc_per_class = {}
    for cls in range(num_classes):
        acc = correct[cls] / total[cls] if total[cls] > 0 else 0.0
        acc_per_class[cls] = acc
    return acc_per_class

### 가중치 자동 보정
# 추가 필요:
import torch.nn.functional as F

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.class_weights = None  # 업데이트됨

    def update_weights(self, classwise_acc):
        # accuracy 낮을수록 weight 높게 (1 - acc)
        weights = [1.0 - acc for acc in classwise_acc.values()]
        self.class_weights = torch.tensor(weights, dtype=torch.float32).cuda()

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1))
        pt = pt.gather(1, target.unsqueeze(1))

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.class_weights is not None:
            weights = self.class_weights.gather(0, target)
            loss = loss * weights.unsqueeze(1)
        return loss.mean()

def evaluate(model, dataloader, device = None, mode = 'fusion'):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for video, keypoint, label in dataloader:
            video = video.to(device)
            keypoint = keypoint.to(device)
            label = label.to(device)

            out = model(video, keypoint)
            preds = torch.argmax(out, dim=1)

            correct += (preds == label).sum().item()
            total += label.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    acc = correct / total
    print(f"### Accuracy: {acc * 100:.2f}%")
    return all_preds, all_labels

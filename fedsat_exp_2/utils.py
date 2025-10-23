
import torch
import time

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

def now_sec():
    return time.time()

def accuracy(logits, targets):
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        return (pred == targets).float().mean().item()

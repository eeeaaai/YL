
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class LogisticMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(28*28, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def resnet18_cifar10(num_classes=10):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, num_classes)
    return m

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sg = 0.3 * torch.clamp(1.0 - x.abs(), min=0.0)
        return grad_output * sg

def spike(x): return SurrogateSpike.apply(x)

class LIFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, leak=0.95):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.leak = leak
    def forward(self, x_t, v):
        v = self.leak * v + self.lin(x_t)
        s = spike(v)
        v = v - s.detach() * v.detach()
        return s, v

class TinySNN_MNIST(nn.Module):
    def __init__(self, steps=20, leak=0.95, num_classes=10):
        super().__init__()
        self.steps = steps
        self.enc = nn.Linear(28*28, 256)
        self.lif1 = LIFLayer(256, 256, leak=leak)
        self.readout = nn.Linear(256, num_classes)
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        v1 = x.new_zeros(B, 256)
        out_sum = x.new_zeros(B, 10)
        for _ in range(self.steps):
            h = torch.relu(self.enc(x))
            s1, v1 = self.lif1(h, v1)
            out = self.readout(s1)
            out_sum = out_sum + out
        return out_sum / float(self.steps)

class TinySNN_CIFAR10(nn.Module):
    def __init__(self, steps=20, leak=0.95, num_classes=10):
        super().__init__()
        self.steps = steps
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
#        self.fc = nn.Linear(64*8*8, 256)
        self.fc = nn.Linear(64*16*16, 256)

        self.readout = nn.Linear(256, num_classes)
        self.lif = LIFLayer(256, 256, leak=leak)
    def forward(self, x):
        B = x.size(0)
        v = x.new_zeros(B, 256)
        out_sum = x.new_zeros(B, 10)
        for _ in range(self.steps):
            h = F.relu(self.conv1(x))
            h = self.pool(F.relu(self.conv2(h)))
            h = h.view(B, -1)
            h = F.relu(self.fc(h))
            s, v = self.lif(h, v)
            out = self.readout(s)
            out_sum = out_sum + out
        return out_sum / float(self.steps)

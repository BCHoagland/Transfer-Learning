import torch
import numpy as np
import math, random

class Env():
    def new(self):
        self.a = random.uniform(0.1, 5.0)
        self.b = random.uniform(0, 2 * math.pi)

    def f(self, x):
        return self.a * math.sin(x + self.b)

    def random_point(self):
        x = random.uniform(-5, 5)
        return (x, self.f(x))

    def sample_points(self, p):
        return [self.random_point() for _ in range(p)]

    def get_loss(self, φ, points):
        x, y = [torch.FloatTensor(p).unsqueeze(0).transpose(0, 1) for p in zip(*points)]
        preds = φ(x)
        loss = torch.pow(preds.squeeze() - y.squeeze(), 2).mean()
        return loss

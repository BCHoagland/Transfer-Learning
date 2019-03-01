from visdom import Visdom
import numpy as np
import torch

viz = Visdom()

losses = []
def plot_loss(step, loss, title):
    if step == 0:
        del losses[:]
    losses.append(loss)

    viz.line(
        X=np.arange(1, len(losses)+1),
        Y=np.array(losses),
        win=title,
        opts=dict(
            title=title,
            ytickmax=2
        )
    )

def plot_f(env, φ, φ2):
    f = np.array([env.f(x) for x in np.arange(-5, 5, 0.1)])
    phi = np.array([φ(torch.FloatTensor([x])).item() for x in np.arange(-5, 5, 0.1)])
    phi2 = np.array([φ2(torch.FloatTensor([x])).item() for x in np.arange(-5, 5, 0.1)])
    points = np.concatenate((np.expand_dims(f, axis=0), np.expand_dims(phi2, axis=0), np.expand_dims(phi, axis=0)), axis=0).transpose()

    title = 'Function'
    viz.line(
        X=np.arange(-5, 5, 0.1),
        Y=points,
        win=title,
        opts=dict(
            title=title,
            legend=['Actual', 'Inner Network', 'Meta Network'],
        )
    )

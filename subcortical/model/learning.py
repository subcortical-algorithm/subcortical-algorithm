import numpy as np
import torch
import pdb


def toCUDA(np_array):
    return torch.from_numpy(np_array.astype(np.float32)).cuda()


class Learner:
    def __init__(self, reservoir, dm, target, lr, learning_rule="force"):
        self.reservoir = reservoir
        self.dm = dm
        self.n = dm.n_class
        self.learning_rule = learning_rule
        self.target_on_ffcurrent = target.on_ffcurrent

        if self.dm.concatenate_input:
            input_num = np.prod(self.reservoir.N0)
        else:
            input_num = 0

        if self.learning_rule == "force":
            if self.dm.connect_layer is None:
                _pn = np.sum(self.reservoir.N)+input_num
            else:
                _pn = np.sum([self.reservoir.N[l] for l in self.dm.connect_layer])+input_num

            self.P = lr * np.eye(_pn)
            self.P = toCUDA(self.P)

    def learn(self, target):
        with torch.no_grad():
            if self.learning_rule == "force":
                self.force_learning(target)

    def force_learning(self,  target):
        reservoir_activity = self.dm.reservoir_activity

        batch_size = reservoir_activity.shape[0]

        I = torch.matmul(reservoir_activity, self.dm.Wout)

        if self.target_on_ffcurrent:
            err = I - target
        else:
            xout = torch.matmul(self.dm.s, self.dm.J) + self.dm.I0 + I
            err = xout - target

        k_fenmu = torch.matmul(reservoir_activity, self.P)
        rPr = torch.sum(k_fenmu * reservoir_activity, dim=1, keepdim=True)
        
        k_fenzi = 1.0 /(1.0 + rPr)
        k = k_fenmu * k_fenzi

        kall = k[:, :, None].repeat(1, 1, self.n)

        dw = -kall * err[:, None, :]

        self.dm.Wout = self.dm.Wout + torch.mean(dw, 0)
        self.P = self.P - torch.matmul(k.transpose(1,0), k_fenmu)/batch_size

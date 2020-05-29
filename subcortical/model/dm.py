from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch

import pdb


def toCUDA(np_array):
    return torch.from_numpy(np_array.astype(np.float32)).cuda()

class ReadoutLayer(ABC):
    def __init__(self, n):
        self.n_class = n
        self.batch_size = 1
        self.reservoir = None
        self.Wout = None
        self.connect_layer=None
        self.concatenate_input = False
        super().__init__()

    def connect_reservoir(self, RN_object, layer=None, concatenate_input=False):
        self.reservoir = RN_object
        self.concatenate_input = concatenate_input
        self.connect_layer = layer

        if concatenate_input:
            input_num = np.prod(self.reservoir.N0)
        else:
            input_num = 0

        if layer is None:
            self.Wout = toCUDA(np.zeros([np.sum(RN_object.N)+input_num, self.n_class]))
        else:
            self.Wout = toCUDA(np.zeros([np.sum([RN_object.N[l] for l in layer])+input_num, self.n_class]))

    def update_from_reservoir(self):
        I = torch.matmul(self.reservoir_activity, self.Wout)
        self.update(I)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.reset()

    @abstractmethod
    def update(self, I):
        pass

    @abstractmethod
    def get_winner(self, average_before_last=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def reservoir_activity(self):
        if self.connect_layer is None:
            _activity = self.reservoir.x_all
        else:
            _activity = torch.cat([self.reservoir.x_all[l] for l in self.connect_layer], dim=1)

        if self.concatenate_input:
            _activity = torch.cat([_activity, self.reservoir.inputs.reshape(self.reservoir.inputs.shape[0], -1)], dim=1)

        return _activity


class TargetFunction:
    def __init__(self, dm, config):
        if config["type"] != "ffcurrent":
            self.alpha = dm.alpha
            self.beta = dm.beta
            self.gamma = dm.gamma
            self.theta = dm.theta
            gb = self.gamma/self.beta
            self.on_ffcurrent = False
            bias = config["bias"]
        
        self.n = dm.n_class    
        self.slop = 2.5
        self.total_step = config["total_step"]
        self._x = np.linspace(-self.slop, self.slop, self.total_step)

        if config["type"] == "classical":
            self.i_correct = dm.Je*(np.tanh(self._x)+1)/2+dm.I0+bias
            self.i_wrong = (dm.Jm+dm.I0-bias)*np.ones(len(self._x))-bias
        elif config["type"] == "modified":
            self.i_correct = dm.Je*(np.tanh(self._x)+1)/2+dm.I0+bias
            self.i_wrong = dm.Jm*(np.tanh(self._x)+1)/2+dm.I0-bias
        elif config["type"] == "constant":
            self.i_correct = (dm.Je+dm.I0+bias) * np.ones(len(self._x))
            self.i_wrong = (dm.Jm+dm.I0-bias) * np.ones(len(self._x))
        elif config["type"] == "ffcurrent": # target on feedforward current
            self.i_correct = 0.1 * np.ones(len(self._x))
            self.i_wrong = -0.1 * np.ones(len(self._x))
            self.on_ffcurrent = True
        else:
            raise Exception("Unsupported target type.")


    def target_function(self, label):
        batch_size = len(label)
        target_curve = np.zeros([batch_size, self.n, self.total_step])
        for i in range(batch_size):
            target_curve[i, :, :] = self.i_wrong
            target_curve[i, label, :] = self.i_correct

        target_curve = toCUDA(target_curve)

        return target_curve


class DecisionMakingConfig:
    def __init__(self, config):
        self.config = config
        n = config["n"]
        self._dm = DecisionMaking(n=n)
        self._dm.__dict__.update(config)
        self.__dict__.update(self._dm.__dict__)

    def connect_reservoir(self, RN_object, layer=None, concatenate_input=True):
        self._dm.connect_reservoir(RN_object, layer=layer, 
                                   concatenate_input=concatenate_input)
        self.__dict__.update(self._dm.__dict__)

    def update_from_reservoir(self):
        self._dm.update_from_reservoir()
        self.__dict__.update(self._dm.__dict__)

    def update(self, I):
        self._dm.update(I)
        self.__dict__.update(self._dm.__dict__)

    def get_state(self):
        state = self._dm.get_state()
        self.__dict__.update(self._dm.__dict__)
        return state

    def get_winner(self, average_before_last=None):
        result = self._dm.get_winner(average_before_last=average_before_last)
        self.__dict__.update(self._dm.__dict__)
        return result

    def get_decision_time(self):
        decision_time = self._dm.get_decision_time()
        self.__dict__.update(self._dm.__dict__)
        return decision_time

    def reset(self):
        self._dm.reset()
        self.__dict__.update(self._dm.__dict__)

    def set_batch_size(self, batch_size):
        self._dm.set_batch_size(batch_size)
        self.__dict__.update(self._dm.__dict__)

    def plot_activity(self, label_ind=None):
        self._dm.plot_activity(label_ind=label_ind)
        self.__dict__.update(self._dm.__dict__)


    def plot_input(self, label_ind=None):
        self._dm.plot_input(label_ind=label_ind)
        self.__dict__.update(self._dm.__dict__)

    @property
    def J(self):
        return self._dm.J


class LinearBaseline(ReadoutLayer):
    def __init__(self, n):
        super(LinearBaseline, self).__init__(n)
        self.voting = np.zeros([self.batch_size, self.n_class])

    def update(self, I):
        _vote = I.argmax(axis=1).cpu()
        for i in range(self.batch_size):
            self.voting[i][_vote[i]] += 1

    def get_winner(self, average_before_last=None):
        result = []
        for i in range(self.batch_size):
            result.append(self.voting[i].argmax())

        return result

    def reset(self):
        self.voting = np.zeros([self.batch_size, self.n_class])


class DecisionMaking(ReadoutLayer):
    def __init__(self, n):
        super(DecisionMaking, self).__init__(n)
        self.alpha = 1.5
        self.theta = 6
        self.beta = 4
        self.gamma = 0.1
        self.tau = 100
        self.dt = 1
        self.Je = 8
        self.Jm = -2
        self.threshold = 0
        self.I0 = 0.7

        self.x = toCUDA(np.zeros([self.batch_size, self.n_class]))
        self.s = toCUDA(np.zeros([self.batch_size, self.n_class]))
        self.r = toCUDA(np.zeros([self.batch_size, self.n_class]))
        self.decision_time = []
        self.r_history = []
        self.x_history = []
        self.i_history = []
        self.t = 0


    def update(self, I, reset_mode=False):
        with torch.no_grad():
            if reset_mode:
                I = toCUDA(np.zeros([self.batch_size, 1]))

            self.x = torch.matmul(self.s, self.J) + self.I0 + I
            self.r = self.beta/self.gamma * torch.log(1+torch.exp((self.x-self.theta)/self.alpha))

            ds = (-self.s + self.gamma*(1-self.s) * self.r) * self.dt / self.tau
            self.s += ds

            if not reset_mode:
                self.i_history.append(I)
                self.x_history.append(self.x)
                self.r_history.append(self.r)
                self.t += 1

                self.decision_time = (torch.max(self.r, dim=1)[0] >= self.threshold).float() * (self.t+1) - 1


    def get_state(self):
        assert self.n_class == 2 and self.batch_size == 1
        if self.r[0, 0] > self.threshold and self.r[0, 1] > self.threshold:
            return "EAS"
        if self.r[0, 0] < self.threshold and self.r[0, 1] < self.threshold:
            return "LAS"
        return "DMS"

    def plot_activity(self, label_ind=None):
        batch_size = self.batch_size
        _r_history = torch.cat(self.r_history, dim=1).reshape(batch_size, -1, self.n_class).cpu().numpy()
        for i in range(batch_size):
            plt.subplot(batch_size, 1, i+1)
            r = _r_history[i, :, :]
            for j in range(self.n_class):
                if j == label_ind[i]:
                    plt.plot(r[:, j], label="correct")
                else:
                    plt.plot(r[:, j])
            plt.legend(loc='upper left')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()


    def plot_input(self, label_ind=None):
        _i_history = torch.cat(self.i_history, dim=1).reshape(self.batch_size, -1, self.n_class).cpu().numpy()
        for i in range(self.batch_size):
            plt.subplot(self.batch_size, 1, i+1)
            inp = _i_history[i, :, :]
            for j in range(self.n_class):
                if j == label_ind[i]:
                    plt.plot(inp[:, j], alpha=0.7, label="correct", linewidth=3)
                else:
                    plt.plot(inp[:, j], alpha=0.7)

            plt.legend(loc='upper left')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()


    def get_winner(self, average_before_last=None):
        result = []
        for i in range(self.batch_size):
            if torch.any(self.r[i,:] > self.threshold):
                result.append(torch.argmax(self.r[i,:]).cpu().numpy()[None][0])
            else:
                result.append(-1)

        return result


    def reset(self):
        self.x = toCUDA(np.zeros([self.batch_size, self.n_class]))
        self.s = toCUDA(np.zeros([self.batch_size, self.n_class]))
        self.r = toCUDA(np.zeros([self.batch_size, self.n_class]))
        self.decision_time = []
        self.x_history = []
        self.r_history = []
        self.i_history = []
        self.t = 0

        # run dynamics for a few steps to get to steady state
        for i in range(5):
            self.update(I=None, reset_mode=True)
            

    def get_decision_time(self):
        decision_time = []
        select_ind = self.get_winner()
        for i in range(self.batch_size):
            _r_history = torch.cat(self.r_history, dim=1).reshape(self.batch_size, -1, self.n_class).cpu().numpy()
            i_r_history_np = _r_history[i, :, :].T
            if np.any(i_r_history_np > self.threshold):
                _dt = np.argmax(i_r_history_np[select_ind[i]] > self.threshold)
                decision_time.append(_dt)

        return decision_time


    @property
    def J(self):
        _J = np.ones([self.n_class, self.n_class]) * self.Jm
        _J = _J - np.eye(self.n_class) * self.Jm + np.eye(self.n_class) * self.Je
        return toCUDA(_J)
    

from tqdm import tqdm

import torch
import sys

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from model.reservoir import ReservoirConfig
from model.dm import DecisionMakingConfig, TargetFunction, LinearBaseline
from model.learning import Learner
from utils.utils import mkdir
# from utils.utils import WriteStream


import pdb

def looper(itera, verbose):
    if verbose:
        return tqdm(itera, leave=False, ncols=80)
    else:
        return itera


def count_correct(prediction, label_ind):
    _correct = 0
    for p, l in zip(prediction, label_ind):
        if p == l:
            _correct += 1
    return _correct



def label2label_ind(class_list, label):
    return [class_list.index(l) for l in label]


class Engine:
    ''' A class that encapsulate the algorithm. 
    
    '''
    def __init__(self, protocol, reservoir_config, dm_config, target_config, dataset, voting=False, queue=None):
        self.protocol = protocol
        self.reservoir_config = reservoir_config
        self.dm_config = dm_config
        self.target_config = target_config
        self.dataset = dataset
        self.reservoir_net = None
        self.decision_making = None
        self.voting = voting
        self.queue = queue
        self.noise_flag = 0


    def __run_model__(self, dataset, batch_size=None, train_flag=True, verbose=False, save_activity=False, dest=None):
        n_correct = 0
        decision_time = []

        if batch_size:
            self.reservoir_net.set_batch_size(batch_size)
            self.decision_making.set_batch_size(batch_size)

        if hasattr(self.reservoir_net, "_reservoir"):
            reservoir_proxy = self.reservoir_net._reservoir
        else:
            reservoir_proxy = self.reservoir_net

        if train_flag:
           reservoir_proxy.noise_level = 0
        else:
           reservoir_proxy.noise_level = self._noise_lv

        # print("Noise level:", self.reservoir_net._reservoir.noise_level)

        for data, label, _ in looper(dataset, verbose):
            if not batch_size:
                self.reservoir_net.set_batch_size(data.shape[0])
                self.decision_making.set_batch_size(data.shape[0])

            label_ind = label2label_ind(self.class_list, label)
            data = data.cuda()

            if train_flag:
                target_curve = self.target.target_function(label=label_ind)

            for t in range(50):
                frame = data[:, t, :, :].reshape(data.shape[0], -1)

                self.reservoir_net.forward(frame)
                self.decision_making.update_from_reservoir()

                if train_flag:
                    self.learner.learn(target_curve[:, :, t])

            prediction = self.decision_making.get_winner(average_before_last=None)
            n_correct = n_correct + count_correct(prediction, label_ind)

            i_decision_time = self.decision_making.get_decision_time()
            decision_time += i_decision_time

            self.reservoir_net.reset()
            self.decision_making.reset()

        if save_activity:


            print("Saving visualization results.")

            self.visualization(dataset.dataset, dest, verbose)

            # plt.hist(decision_time, color="#3F5D7D", bins=np.arange(1, 51, 1), density=True)
            # plt.xticks(fontsize=14)
            # plt.yticks(fontsize=14)
            # plt.ylim([0, 0.4])
            # plt.xlabel("Decision ", fontsize=16)
            # plt.ylabel("Count", fontsize=16)
            # plt.grid(True)

            # plt.savefig(osp.join(dest, 'decision_time.png'))
            # plt.close()

        acc = n_correct / dataset.dataset.total

        return acc, decision_time


    def run(self, verbose=False, save_activity=False, dest=None):
        if not dest:
            dest = '.'

        # if self.queue:
        #     sys.stdout = WriteStream(self.queue)

        # Parameters
        self.n_class = self.dm_config["n"]
        n_epoch = self.protocol["n_epoch"]
        n_train = self.protocol["n_train"]
        n_val = self.protocol["n_val"]
        T = self.target_config["total_step"]
        learning_rate = self.target_config["learning_rate"]
        time = np.arange(0, T, 1)

        # Model instantiation
        self.reservoir_net = ReservoirConfig(self.reservoir_config)
        if self.voting:
            self.decision_making = LinearBaseline(self.dm_config["n"])
            dm2learner = self.decision_making

            print("Using linear baseline model.")

        else:
            self.decision_making = DecisionMakingConfig(self.dm_config)
            dm2learner = self.decision_making._dm
        self.decision_making.connect_reservoir(self.reservoir_net, concatenate_input=True)
        self.target = TargetFunction(self.decision_making, self.target_config)

        self.learner = Learner(self.reservoir_net, dm2learner, self.target,
                          lr=learning_rate, learning_rule="force")

        self.train_set = self.dataset["train"]
        self.val_set = self.dataset["validate"]
        self.test_set = self.dataset["test"]
        self.class_list = self.train_set.dataset.class_list

        self._noise_lv = self.reservoir_net.noise_level

        best_val_acc = 0
        best_Wout = None
        i_train = 0


        for i_epoch in range(n_epoch):
            if verbose:
                print("==== Running on Epoch %d/%d. ===="%(i_epoch+1, n_epoch))

            t_save_activity = save_activity and i_epoch == n_epoch-1

            # Training
            if verbose:
                print("Training on dataset.")
            acc, dt_train = self.__run_model__(self.train_set, 
                                               train_flag=True, 
                                               batch_size=1, 
                                               verbose=verbose, 
                                               save_activity=t_save_activity, 
                                               dest=osp.join(dest, 'train'))

            
            if verbose:
                print("Training accuracy: %.4f" % acc)

            i_train += 1

            # Validation
            if (i_train == n_train or i_epoch == n_epoch-1) and n_val > 0:
                if verbose:
                    print("Validating on dataset.")

                val_acc, dt_val = self.__run_model__(self.val_set,
                                                     train_flag=False,
                                                     batch_size=None,
                                                     verbose=verbose,
                                                     save_activity=False,
                                                     dest=None)

                if verbose:
                    print("Validation accuracy: %.4f" % (val_acc))

                if best_val_acc < val_acc:
                    best_Wout = self.decision_making.Wout
                    best_val_acc = val_acc
                    if verbose:
                        print("Saved model with best validation accuracy.")

                i_train = 0

        # Testing
        if best_Wout is not None:
            if verbose:
                print("Testing using Wout from validation with best accuracy:%.4f" % (best_val_acc))
            dm2learner.Wout = best_Wout
            self.decision_making.Wout = best_Wout

        test_acc, dt_test = self.__run_model__(self.test_set,
                                               train_flag=False,
                                               batch_size=None,
                                               verbose=verbose,
                                               save_activity=save_activity,
                                               dest=osp.join(dest, 'test'))


        print("Testing accuracy: %.4f" % (test_acc))


        return test_acc

    def visualization(self, dataset, dest, verbose=False):

        for ind in looper(range(self.n_class), verbose):
            # Only visualize the first example for each subject for the sake of performance
            data, label, data_path = dataset[ind*dataset.n_example]
            self.reservoir_net.set_batch_size(1)
            self.decision_making.set_batch_size(1)

            label_ind = label2label_ind(self.class_list, [label])
            target_curve = self.target.target_function(label=label_ind)
            data = torch.from_numpy(data).cuda()

            mkdir(osp.join(dest, *data_path.split("/")[-2:], "r_frames"))
            mkdir(osp.join(dest, *data_path.split("/")[-2:], "i_frames"))


            for t in range(50):
                frame = data[None, t, :, :].reshape(1, -1)
                self.reservoir_net.forward(frame)
                self.decision_making.update_from_reservoir()

                _r_history = torch.cat(self.decision_making.r_history, dim=1).reshape(1, -1, self.n_class).cpu().numpy()

                r = _r_history[0, :, :]
                for j in range(self.n_class):
                    if j == label_ind[0]:
                        plt.plot(r[:, j], linewidth=2, linestyle="dashed", label="correct")
                    else:
                        plt.plot(r[:, j])
                plt.xlim([0, 50])
                plt.ylim([0, 100])
                plt.xlabel("Input Sequence Index")
                plt.ylabel("Activity")
                plt.legend(loc='upper left')
                plt.title("Decision-Making Neuron Activity")
                plt.grid(True)
                
                plt.savefig(osp.join(osp.join(dest, *data_path.split("/")[-2:]), "r_frames", str(t+1).zfill(3)+".png"))
                plt.close()


                # _i_history = torch.cat(self.decision_making.i_history, dim=1).reshape(1, -1, self.n_class).cpu().numpy()
                # i = _i_history[0, :, :]
                # for j in range(self.n_class):
                #     if j == label_ind[0]:
                #         plt.plot(i[:, j], linewidth=3, alpha=0.7, linestyle="dashed", label="correct")
                #         plt.plot(np.mean(i[:, j])*np.ones([i.shape[0], 1]), linewidth=3, linestyle="dashed", label="correct (mean)")
                #     else:
                #         plt.plot(i[:, j], alpha=0.3)
                # plt.xlim([0, 50])
                # plt.ylim([-0.15, 0.15])
                # plt.xlabel("Input Sequence Index")
                # plt.ylabel("Input current")
                # plt.legend(loc='upper left')
                # plt.grid(True)
                
                # plt.savefig(osp.join(osp.join(dest, *data_path.split("/")[-2:]), "i_frames", str(t+1).zfill(3)+".png"))
                # plt.close()

            self.reservoir_net.reset()
            self.decision_making.reset()


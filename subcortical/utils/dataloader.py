import numpy as np
import torch
import os
import cv2

import os.path as osp

from torch.utils.data import Dataset, DataLoader

RESIZE_H = 72
RESIZE_W = 96


def remove_element(original_list, remove_list):
    for _e in remove_list:
        original_list.remove(_e)
    return original_list

def get_border(frame):
    h, w = frame.shape
    _t = np.max(frame, axis=1)

    head_loc = next(x[0] for x in enumerate(_t) if x[1] > 0.1)
    feet_loc = h-next(x[0] for x in enumerate(reversed(_t)) if x[1] > 0.1)
    return feet_loc, head_loc

def has_content(frame):
    return False if np.sum(frame) < 1 else True

def centering_person(sequence):
    T, h, w = sequence.shape
    for i in range(T):
        frame = sequence[i]
        if has_content(frame):
            center_x = np.sum(frame * np.arange(w)) / np.sum(frame)
            n_shift_right = int(w/2 - center_x)
            frame = np.roll(frame, n_shift_right, axis=1)

            feet_loc, _ = get_border(frame)
            n_shift_down = int(h*0.9 - feet_loc)
            frame = np.roll(frame, n_shift_down, axis=0)

        sequence[i] = frame

    return sequence


class GaitData(Dataset):
    def __init__(self, path, mode, n_class=5, n_example=5, 
                 ref_dataset=None, class_list=None, centering=True):
        assert mode == "train" or mode == "validate" or mode == "test"
        self.mode = mode
        self.n_class = n_class
        self.n_example = n_example
        self.path = path
        self.people_list = os.listdir(path)
        self.people_list = [x for x in self.people_list if x[0] != '.']

        self.ref_dataset = None
        self.centering = centering

        self.n = len(self.people_list)
        self.n_trial = 50
        self.n_step = 50
        self.h = RESIZE_H
        self.w = RESIZE_W

        if class_list is None:
            self.class_list = np.random.choice(self.n, self.n_class, replace=False).tolist()
        else:
            self.class_list = class_list

        if ref_dataset is not None:
            self.ref_dataset = ref_dataset
            self.class_list = self.ref_dataset[0].class_list

        self.example_ind = []

        # Select DIFFERENT trial indices for DIFFERENT identities.

        for ind in range(self.n_class):
            if self.ref_dataset is None:
                select_ind = np.random.choice(self.n_trial, self.n_example, replace=False).tolist()
            else:
                _used = []
                for _ref in ref_dataset:
                    _used = _used + _ref.example_ind[ind]
                _candidate = remove_element(np.arange(self.n_trial).tolist(), _used)
                select_ind = np.random.choice(_candidate, self.n_example, replace=False).tolist()

            select_ind.sort()
            self.example_ind.append(select_ind)

        self.total = self.n_class*self.n_example

    def __getitem__(self, ind):
        sub_ind = int(ind/self.n_example)
        exp_ind = ind-sub_ind*self.n_example

        data, data_path = read_data_from_disk(self.path, self.people_list, self.class_list[sub_ind], self.example_ind[sub_ind][exp_ind])

        if self.centering:
            data = centering_person(data)

        _mean = np.mean(data)
        _std = np.std(data)
        if _std > 0.01:
            data = (data-_mean) / _std
        
        label = self.class_list[sub_ind]
        return data, label, data_path

    def __len__(self):
        return self.total


def read_data_from_disk(path, whole_set, selected_people, selected_trial):
    trial_list = os.listdir(osp.join(path, whole_set[selected_people]))
    trial_list = [x for x in trial_list if x[0] != '.']
    trial = trial_list[selected_trial]
    frame_list = os.listdir(osp.join(path, whole_set[selected_people], trial))     # files
    frame_list = [x for x in frame_list if x[0] != '.' and x[-4:]==".jpg"]
    frame_list.sort()
    trial_data = []

    for frame in frame_list:
        img = cv2.imread(osp.join(path, whole_set[selected_people], trial, frame))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize_gray = cv2.resize(gray, (RESIZE_W, RESIZE_H))
        trial_data.append(resize_gray)

    if len(trial_data) < 50:
        for i in range(50-len(trial_data)):
            trial_data.append(np.zeros([RESIZE_H, RESIZE_W]).astype(np.uint8))

    trial_data = np.array(trial_data).squeeze().astype(np.float32)

    return trial_data, osp.join(path, whole_set[selected_people], trial)


def get_dataset(npy_path, n_class=5, n_train=5, n_validate=15, n_test=35, 
                class_list=None, train_batch=1, val_batch=10, test_batch=10,
                centering=True):

    train_set = GaitData(npy_path, mode="train", n_class=n_class, n_example=n_train, 
                         class_list=class_list, centering=centering)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=train_batch, num_workers=0)

    validate_set = GaitData(npy_path, mode="validate", n_class=n_class, n_example=n_validate, 
                         ref_dataset=[train_set], centering=centering)
    validate_loader = DataLoader(validate_set, shuffle=True, batch_size=val_batch, num_workers=0)

    test_set = GaitData(npy_path, mode="test", n_class=n_class, n_example=n_test, 
                         ref_dataset=[train_set, validate_set], centering=centering)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=test_batch, num_workers=0)

    return {"train":train_loader, "validate":validate_loader, "test":test_loader}



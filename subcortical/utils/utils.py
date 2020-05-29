# from PyQt5 import QtCore
# from PyQt5.QtCore import QObject, pyqtSlot
from tqdm import tqdm

import os
import datetime
import subprocess

import os.path as osp


# class WriteStream(object):
#     def __init__(self,queue):
#         self.queue = queue

#     def write(self, text):
#         self.queue.put(text)


# class MyReceiver(QObject):
#     mysignal = QtCore.pyqtSignal(str)

#     def __init__(self,queue,*args,**kwargs):
#         QObject.__init__(self,*args,**kwargs)
#         self.queue = queue

#     @pyqtSlot()
#     def run(self):
#         while True:
#             text = self.queue.get()
#             if len(text) <3:
#             	continue
#             currentDT = datetime.datetime.now()
#             self.mysignal.emit('['+currentDT.strftime("%Y-%m-%d %H:%M:%S")+']'+" "*3+text+"\n")


def mkdir(name):
	if not os.path.exists(name):
		os.makedirs(name)


def make_movie(activity_root, vis_path_list=None):
    ''' A helper function calling ffmpeg to make movies by combining input data and
    the model results for better visualization. '''
    assert isinstance(vis_path_list, list)

    fr = "12"
    if not activity_root:
        activity_root = "."
    subject_list = os.listdir(activity_root)
    subject_list = [x for x in subject_list if x[0] != "_" and x[0] != "."]
    subject_list = [x for x in subject_list if osp.isdir(osp.join(activity_root, x))]

    for subject in subject_list:
        trial_list = os.listdir(osp.join(activity_root, subject))
        trial_list = [x for x in trial_list if x[0] != "_" and x[0] != "."]

        n_i = 0

        for trial in trial_list:
            input_path = []
            for vis_pi in vis_path_list:
                if vis_pi:
                    input_path.append(osp.join(vis_pi, subject, trial, trial+"-%3d.jpg"))
                    n_i += 1

            activity = osp.join(activity_root, subject, trial, "r_frames", "%3d.png")
            n_i += 1
            # current = osp.join(activity_root, subject, trial, "i_frames", "%3d.png")
            # n_i += 1
            output = osp.join(activity_root, subject, trial, "visualization_result.mp4")

            config = ["ffmpeg", "-r", fr]

            for ind, vis_pi in enumerate(vis_path_list):
                if vis_pi:
                    config += ["-i", input_path[ind], "-r", fr]


            config += ["-i", activity, "-r", fr]
            # config += ["-i", current, "-r", fr]

            config += ["-vcodec", "mpeg4", "-b", "10000k",
                       "-filter_complex", "hstack=inputs="+str(n_i), output]

            subprocess.run(config, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

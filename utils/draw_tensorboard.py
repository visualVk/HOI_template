from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import threading
import shutil
from torch.utils.tensorboard import SummaryWriter
from config.config import config


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class TensorWriter(object):
    def __init__(self):
        # whether log dir is existed or not
        if not os.path.exists(config.LOG_DIR):
            os.mkdir(config.LOG_DIR)

        # create log dir
        new_file = os.path.join(
            config.LOG_DIR, config.PROJECT_NAME + str(time.time()))
        os.mkdir(new_file)

        # move logs remained to file named '{project_name + timestamp}'
        for file in os.listdir(config.LOG_DIR):
            if os.path.isdir(os.path.join(config.LOG_DIR, file)):
                continue

            t = threading.Thread(target=shutil.move, args=[os.path.join(
                config.LOG_DIR, file), new_file])
            t.start()
            t.join()

        self.writer = SummaryWriter(
            config.LOG_DIR, comment=config.PROJECT_NAME)

    @property
    def writer(self) -> SummaryWriter:
        return self.writer

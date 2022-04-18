from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class TensorWriter(object):
    def __init__(self, config=None):
        assert config is not None, "singleton(TensorWriter) should be passed config in the first time"
        # whether log dir is existed or not
        if not os.path.exists(config.LOG_DIR):
            os.mkdir(config.LOG_DIR)

        # create log dir
        dir_name = os.path.join(
            config.LOG_DIR,
            f"{config.PROJECT_NAME}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.mkdir(dir_name)

        # move logs remained to file named '{project_name + timestamp}'
        # for file in os.listdir(config.LOG_DIR):
        #     if os.path.isdir(os.path.join(config.LOG_DIR, file)):
        #         continue
        #
        #     t = threading.Thread(target=shutil.move, args=[os.path.join(
        #         config.LOG_DIR, file), new_file])
        #     t.start()
        #     t.join()

        self._writer = SummaryWriter(
            dir_name, comment=config.PROJECT_NAME)

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

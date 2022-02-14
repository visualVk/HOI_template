import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

os.chdir(sys.path[0])

# proj_path = os.join(this_dir, 'lib')
# add_path(proj_path)

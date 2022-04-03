import settings
import os

def clean(path):
    filelist = [f for f in os.listdir(path) if f.endswith('mmap')]
    for f in filelist:
        os.remove(os.path.join(path, f))

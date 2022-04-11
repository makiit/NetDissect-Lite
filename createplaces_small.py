import os
import shutil
import glob
import random
path = "/expanse/lustre/projects/ddp390/makhan1/places365_standard/train"
dirs=os.listdir(path)
dest = "/expanse/lustre/projects/ddp390/makhan1/places_small/train"
for c in dirs:
    print(c)
    s = path+"/%s/*"%c
    f_list = random.sample(glob.glob(s),1000)
#     print(s)
    p = dest+"/%s/"%c
    if not os.path.exists(p):
        os.makedirs(p)
    for f in f_list:
        shutil.copy(f,p)
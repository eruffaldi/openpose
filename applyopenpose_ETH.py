import numpy as np
import subprocess, os

rootdir = "/media/headpose/hpdb"
openpose_exe = "/home/lando/lib/openpose/build/examples/openpose/openpose.bin"
outroot = "/home/lando/Desktop/headpose/hpdb"

for i in range(1,25):
	suffix = str(i).zfill(2)
	curdir = rootdir + "/" + suffix
	curout = outroot + "/" + suffix

	if not os.path.exists(curout):
		os.makedirs(curout)

	subprocess.call([openpose_exe, "--image_dir", curdir, "-write_keypoint", curout])






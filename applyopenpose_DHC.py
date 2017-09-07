import numpy as np
import subprocess, os

rootdir = "/home/lando/Desktop/headpose/ict3DHP_data"
openpose_exe = "/home/lando/lib/openpose/build/examples/openpose/openpose.bin"
outroot = "/home/lando/pelars/headposes/openpose/C++/data/output/ict/"

for i in range(2,21):
	if i % 2 == 0:
		suffix = str(i).zfill(2)
		curdir = rootdir + "/" + suffix + "/colour undist.avi"
		curout = outroot + "/" + suffix

		if not os.path.exists(curout):
			os.makedirs(curout)

		subprocess.call([openpose_exe, "-video", curdir, "-write_keypoint", curout, "-no_display"])






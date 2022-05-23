import os
import sys
import numpy as np
import cv2

def generate_video(img_list,tar_name):

    frame = cv2.imread(img_list[0])
    H,W,C = frame.shape

    _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _out = cv2.VideoWriter(tar_name,_fourcc,30.0,(W,H))
    for filename in img_list:
        frame = cv2.imread(filename)
        _out.write(frame)
    _out.release()

def gen_video_from_seq(root_path):
    #dir_list = next(os.walk(root_path))[1]
    dir_list = [x[0] for x in os.walk(root_path)]
    for i_dir in dir_list:
        #full_path = os.path.join(root_path,i_dir)
        full_path = i_dir
        filename_list = []
        id = 0
        while True:
            filename = os.path.join(full_path,'%05d.png' % id)
            if not os.path.exists(filename):
                break
            filename_list.append(filename)
            id += 1
        if id > 0:
            print('get image sequence %d from %s' % (id, full_path))
            generate_video(filename_list, full_path + ".mp4")

if __name__ == "__main__":
    gen_video_from_seq(sys.argv[1])
import os
import pandas as pd
import pathlib as Path
import cv2
import numpy as np 

input_base = "kaggle dataset deepfake"
output_base = "videoframes"

num_of_frames = 35


for labels in ["real","fake"]:
    input_folder = os.path.join(input_base,labels)
    output_folder = os.path.join(output_base,labels)

    os.makedirs(output_folder,exist_ok=True)
    for video_name in os.listdir(input_folder):
        video_path = os.path.join(input_folder,video_name)
        video_output_path = os.path.join(output_folder,video_name.split()[0])

        os.makedirs(video_output_path,exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_of_frames:
            cap.release()
            continue

        frame_indices = np.linspace(0,total_frames-1,num_of_frames,dtype=int)

        for i,idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ret,frame = cap.read()

            if ret:
                frame = cv2.resize(frame,(224,224))
                cv2.imwrite(os.path.join(video_output_path,f"frame_{i}.jpg"),frame)
            
        cap.release()


print("frame extraction is done successfully")

    



    

        

    
import pandas as pd
import numpy as np
import os
from retinaface import RetinaFace
import tqdm
import cv2

root_input = "videoframes"
root_output = "faceframes"

for label in ["real","fake"]:
    input_folder = os.path.join(root_input,label)
    output_folder = os.path.join(root_output,label)

    os.makedirs(output_folder,exist_ok=True)

    for videos in os.listdir(input_folder):
        input_video_path = os.path.join(input_folder,videos)
        output_video_path = os.path.join(output_folder,videos)

        os.makedirs(output_video_path,exist_ok=True)

        for frame_name in tqdm(os.listdir(input_video_path)):
            frame_path = os.path.join(input_video_path,frame_name)

            img = cv2.imread(frame_path)

            if img is None:
                continue
            
            detections = RetinaFace.detect_faces(frame_path)

            if isinstance(detections,dict):

                first_face = list(detections.values())[0]
                x1,y1,x2,y2 = first_face["facial_area"]

                crop_face = img[y1:y2, x1:x2]
                crop_face = cv2.resize(crop_face,(224,224))

                save_path = os.path.join(output_video_path,frame_name)
                cv2.imwrite(save_path,crop_face)






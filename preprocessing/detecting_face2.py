from concurrent.futures import ProcessPoolExecutor
import os
import cv2
from exceptions.exception import Exception
from loggings.loggers import logging
from retinaface import RetinaFace
from tqdm import tqdm
from functools import partial
import torch
import sys
from ultralytics import YOLO
import numpy as np


model = YOLO("preprocessing\yolov8n-face.pt")

def detect_face(video_name,input_path,output_path):
      try:
            model = YOLO("preprocessing\yolov8n-face.pt")
            
            input_video_path = os.path.join(input_path,video_name)
            output_video_path = os.path.join(output_path,video_name)


            os.makedirs(output_video_path,exist_ok=True)
            for frame_name in os.listdir(input_video_path):
                  frame_path = os.path.join(input_video_path,frame_name)
                  img = cv2.imread(frame_path)

                  if img is None:
                        continue
                  
                  preds = model(img,conf = 0.5 ,verbose = False)

                  for pred in preds:
                        if len(pred.boxes)>0:
                              box = pred.boxes.xyxy[0].cpu().numpy().astype(int)
                              x1, y1, x2, y2 = box
                              face_found = True
                              crop_face = img[y1:y2, x1:x2]
                        
                              if crop_face.size > 0:
                                    crop_face = cv2.resize(crop_face,(224,224))
                                    save_path = os.path.join(output_video_path,frame_name)
                                    cv2.imwrite(save_path,crop_face)
                              break
      except Exception as e:
            raise Exception(e,sys)      


def main():
      try:
            input_folder = "videoframes"
            output_folder = "faceframes"
            
            for label in ["real","fake"]:
                  input_folder_path = os.path.join(input_folder,label)
                  output_folder_path = os.path.join(output_folder,label)

                  os.makedirs(output_folder_path,exist_ok=True)
                  video_folder = os.listdir(input_folder_path)
                  
                  func = partial(detect_face,input_path = input_folder_path,output_path = output_folder_path)

                  with ProcessPoolExecutor(max_workers= os.cpu_count() // 2) as executor:
                        list(tqdm(executor.map(func,video_folder),
                              total=len(video_folder)))
      except Exception as e:
            raise Exception(e,sys)

if __name__ == "__main__":
      main()


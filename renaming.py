import os 
import pandas as pd

real_path = r"kaggle dataset deepfake\real"
fake_path = r"kaggle dataset deepfake\fake"

count = 1

real_videos = os.listdir(real_path)
fake_videos = os.listdir(fake_path)

for file in real_videos:
    if file.endswith(".mp4"):
        old_path = os.path.join(real_path,file)
        new_name = f"video{count}.mp4"
        new_path = os.path.join(real_path,new_name)

        os.rename(old_path,new_path)
        count+=1

count = 1 

for file in fake_videos:
    if file.endswith(".mp4"):
        old_path = os.path.join(fake_path,file)
        new_name = f"fakeVideo{count}.mp4"
        new_path = os.path.join(fake_path,new_name)
        os.rename(old_path,new_path)

        count+=1

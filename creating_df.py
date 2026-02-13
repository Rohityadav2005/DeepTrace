import os
import pandas as pd
from pathlib import Path

real_dir = r"kaggle dataset deepfake\real"
fake_dir = r"kaggle dataset deepfake\fake"

def create_paired_df(real_dir,fake_dir):
    
    pair = []

    original_videos = Path(real_dir).glob("*.mp4")

    for orgin_path in original_videos:
        orgin_name = orgin_path.stem.lower()

        fake_pattern = f"fake{orgin_name}.mp4"

        fake_found = False

        fake_path = Path(fake_dir) /fake_pattern
        if fake_path.exists():
            pair.append({
                "orignal_video_path" : str(orgin_path),
                "fake_video_path"    : str(fake_path),
                "original_label"     : 1,
                "fake_label"         : 0,
                "pair_id"            : orgin_name

            })
            fake_found = True

        if not fake_found:
            fake_path = list(Path(fake_dir).glob(f"*{orgin_name}*.mp4"))
            if fake_path:
                pair.append({
                    "orignal_video_path" : str(orgin_path),
                    "fake_video_path"    : str(fake_path),
                    "original_label"     : 1,
                    "fake_label"         : 0,
                    "pair_id"            : orgin_name

                })
            else:
                print(f"No fake videos were found for {orgin_name}")

    return pd.DataFrame(pair)


df = create_paired_df(real_dir,fake_dir)

print(df.head())



        

    


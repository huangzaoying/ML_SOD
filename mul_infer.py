import os
import subprocess

folder_path = "../SOD/Saliency-TestSet/Stimuli"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            file_path = os.path.join(root, file)
            output_path = os.path.join(
                "./results", "/".join(file_path.split("/")[-2:-1])
            )
            print(file_path, output_path)
            command = f"python inference.py --img_path {file_path} --output_path {output_path}"
            subprocess.run(command, shell=True)

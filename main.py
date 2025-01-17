!nvidia-smi
!pip install ultralytics
from ultralytics import YOLO
import os
from IPython import display
from IPython.display import display, Image
from IPython.display import clear_output

clear_output()
!yolo mode=checks

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="OcXMsJUrph5yedjsNnwf")
project = rf.workspace("roboflow-gw7yv").project("raccoon")
dataset = project.version(2).download("yolov8")

!yolo task=detect mode=train  model=yolov8m.pt data={dataset.location}/data.yaml epochs=20 imgsz=640

Image(filename=f'/content/runs/detect/train3/confusion_matrix.png', width=600)

Image(filename=f'/content/runs/detect/train3/results.png', width=600)

!yolo task=detect mode=val model=/content/runs/detect/train15/weights/best.pt data={dataset.location}/data.yaml

!yolo task=detect mode=predict model=/content/runs/detect/train3/weights/best.pt conf=0.5 source={dataset.location}/test/images

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'/content/runs/detect/predict/*.jpg'):
        display(Image(filename=image_path, height=600))
        print("\n")

# BrainTumorsSegmentation_yolo26
Brain tumor segmentation using yolo26-seg and the Roboflow file https://universe.roboflow.com/mri-brain-tumor/brain-tumor-segmentation-8aek5/dataset/1

Installation:

Download and extract the project folder.

Download the file https://universe.roboflow.com/mri-brain-tumor/brain-tumor-segmentation-8aek5/dataset/1, which requires a Roboflow key that can be obtained for free.

The result is the Brain tumor segmentation.v1i.yolov8-obb ​​folder, which should be placed inside the project folder.

As with any project testing, it is advisable to perform the test in a new environment.

If you have an older version of YOLO, you must update it, otherwise the YOLO26 modules will not be referenced and will not be downloaded automatically when requested.

`pip install -U ultralytics`

Otherwise, you must install it.

`pip install ultralytics`

A requirements.txt file is attached.

EVALUATION:

Running:

`TestBrainTumorsSegmentation_yolo26.py`

Three sample images are included.

Each image shows the image labeled by RoboFlow (rectangles appear because they have been segmented with only four points), which are assumed to correspond to the tumor's location, and the image predicted and segmented by the model.

![Figure 1](https://github.com/ablanco1950/BrainTumorsSegmentation_yolo26/blob/main/Figure_1.png)


![Figure 2](https://github.com/ablanco1950/BrainTumorsSegmentation_yolo26/blob/main/Figure_2.png)


![Figure 3](https://github.com/ablanco1950/BrainTumorsSegmentation_yolo26/blob/main/Figure_3.png)


TRAINING:

The best.pt model was obtained by running the program

TrainBrainTumorsSegmentation_yolo26.py 

With 200 epochs.

In each epoch, the best and latest models are stored in the directory:

runs/segment/entrenamientos_yolo26/segmentacion_custom/weights

A folder named segmentacion_custom is created with each run.

The folder segmentacion_custom-3.zip containing the reports generated during training is included.

The document LOG200epochTrainBrainTumorsSegmentation_yolo26.docx, containing the console messages during training, is also included, resulting in an mAP50 of 0.81 and an mAP50-95 of 0.561.


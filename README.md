# IMAGE-AI_extracting-vehicle-from-image
detecting vehicle from images using YOLOV3([IMAGE-AI repo](https://github.com/OlafenwaMoses/ImageAI))
## Description

> in this project,we have used YOLO-V3 Object Detection Model constructed by `IMAGE-AI` forked by me ,so by using YOLO-V3,i have enabled to exract vehicle in different image dataset,and passing them to CNN model for detecting vehicle color.

## to detect objects
>first,download ([IMAGE-AI repo](https://github.com/OlafenwaMoses/ImageAI)) and put `imageAI.ipynb` in this repo.then make 3 folder `input`,`output`,`output2` in this repo.

>hint:put image dataset in `input` folder.

>in seconde step open `imageAI.ipynb` and run code cell in Colab step by step:
>> 1) mount your google drive and set your path.
```
from google.colab import drive
drive.mount('/content/drive/')
import os
os.chdir("/content/drive/MyDrive/train python/IMAGE-AI/ImageAI-master")
!ls
```
>> 2) install requirements.txt by :
```
!pip install -r requirements.txt
```
>> 3) importing libraries:
```
import imageai
from PIL import Image
import os 
import cv2
from imageai.Detection import ObjectDetection
```
>> 4) set `path` and `directory` basis on own root in `google drive`.
```
path ="/content/drive/MyDrive/train python/IMAGE-AI/ImageAI-master/input"
directory="/content/drive/MyDrive/train python/IMAGE-AI/ImageAI-master/output"
execution_path=os.getcwd()
p=os.listdir(path)
```
>> 5) download YOLO-V3 [weight](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5) in the `IMAGE-AI repo` and put in in the directory,then identify objects that you want to extract from image by setting `True` in object variable.
```
i=1
j=1
img_size=400
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
custom = detector.CustomObjects(car=True, bus= True, truck= True)
```
>> 6) each object in `input` directory was detected ind was set in `output`(all objects in one image) and `output2`(each object in one image).
```
for f in p :
    if f.endswith("jpg"):
        detections =detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=os.path.join(path , f), output_image_path=os.path.join(directory , "newImage"+str(i)+".jpg"), minimum_percentage_probability=30)
        img = cv2.imread(os.path.join(path  , f))
        for eachObject in detections:
            print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
            points = eachObject["box_points"]
            print("--------------------------------")
            points = eachObject["box_points"]
            print("this image is"+ f)
            crop_img = img[points[1]:points[3],points[0]:points[2]]
            cv2.imwrite("/content/drive/MyDrive/train python/IMAGE-AI/ImageAI-master/output2/"+str(j)+".jpg",crop_img)
            j+=1
        i+=1  
```

# Result
>the results in `output` directory are below image:

![newImage1](https://user-images.githubusercontent.com/53394692/110380627-9c7a8d80-806d-11eb-815f-8774ed656aaa.jpg)
>the results in `output2` directory are below images:
















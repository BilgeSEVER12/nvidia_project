# Smoking Detection
When we walk on the street there are a lot of people smoking cigarettes. Smoking cigarettes are not forbidden if you aren't smoking in a public place. Some people smoke in a public area however they are not punished. Because no one sees them while they are smoking. We can solve this problem with AI. If someone holds a cigarette we can find with a camera and punish them for what he/she did. 
## The Algorithm
I followed 2 different methods and datasets to tackle this problem. In both cases I need to classify/detect only one class: smoking. Since none of the available networks has cigarette in their classes, I needed to retrain models using custom datasets. I used both image classification and detection projects, as you can find in the following sections. 
###  Image Classification
At the begining, I used "imageNet" for image recognition : [image recognition](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md) I used pretrained Resnet-18 model to retrain with my custom dataset. 
###  Image Detection
Later, I started to work with "detectNet" for object detection
[object detection](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md) I used SSD algorithm which is supported by jetson-inference. 

In order to use object detection I labeled more than 1000 images containing cigarettes with CVAT tool as proposed by Nvidia. By using CVAT tool: 
1. I created a detection (annotation) [project](https://imgur.com/KlFXHcm)
2. I divided the dataset into three parts: training, validation and test. 
3. I created related annotation jobs. 

It took me a long time to label the cigarettes because there were so many cigarette photos. [Here](https://imgur.com/a/H6zCmYl) is only one example. 
## Running This Project
###  Image Classification
At first, I ran jetson-nano in headless mode then I used the [jetson-inference](https://github.com/dusty-nv/jetson-inference) project to find how  I can do Image Classification. After that I used the [dataset from Kaggle](https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection?resource=download)

Below are the steps I have followed to run this applicaiton: 
* Run docker

`./docker/run.sh`
* Retrain the model

`cd python/training/classification/`

`python3 train.py --model-dir=models/smoke_nonsmoke data/smoke_nonsmoke`
* Convert the model to the best model onnx

`python3 onnx_export.py --model-dir=models/smoke_nonsmoke/`
* I test the model on test images

`NET=models/smoke_nonsmoke`

`DATASET=data/smoke_nonsmoke`

`mkdir $DATASET/test_output_smoke $DATASET/test_output_nonsmoke`

`imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/smoke $DATASET/test_output_smoke`

`imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/nonsmoke $DATASET/test_output_nonsmoke`
### Image Detection 
After I got unsatisfactory result with my classification project I decided to go with image deteciton. I used a different [dataset](https://data.mendeley.com/datasets/7b52hhzs3r/1) this time. Below are the steps I have followed to run this applicaiton: 
* connect over USB (headless mode)
* turn on fan

`sudo sh -c 'echo 128 > /sys/devices/pwm-fan/target_pwm'`

* copy custom dataset in VOC format (root owns data directory! so sudo command worked!)

`cd jetson-inference/python/training/detection/ssd/data`

`sudo mkdir dssmoking`

`cd dssmoking` 

`sudo scp hp@192.168.55.100:/home/hp/Downloads/dssmoking.zip .`

`sudo unzip dssmoking.zip`

* create labels.txt file in both dataset folders. File contains one line "cigarette". 

* download SSD-Mobilenet-v1 (needed due to retrain error! )

`cd tools`

`./download.models.sh`

`wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth`

* run docker

* retrain the model using our own dataset

`cd python/training/detection/ssd`

`python3 train_ssd.py --dataset-type=voc --data data/dssmoking --model-dir=models/modelsmoking --batch-size=2 --workers=1`

* to resume if training stops

`python3 train_ssd.py --dataset-type=voc --data data/dssmoking --model-dir=models/modelsmoking --batch-size=2 --workers=1 --resume models/modelsmoking/mb1-ssd-Epoch-0-Loss-7.829309706785241.pth`

* export to ONNX, it selects the best checkpoint with minimum lost. 

`python3 onnx_export.py --model-dir=models/model1`

* run inference. use labels.txt created under modelsmoking folder after training completes.  

`cd jetson-inference/python/training/detection/ssd`

`NET=models/modelsmoking`

`IMAGES=/home/nvidia/Pictures`

`PYTORCH_MODEL=mb1-ssd-Epoch-0-Loss-7.829309706785241.pth`

`PYTORCH_MODEL=mb1-ssd-Epoch-0-Loss-9.759434202442998.pth`

* pytorch model test, under docker
Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>

           `python run_ssd_example.py mb1-ssd  $NET/$PYTORCH_MODEL $NET/labels.txt data/dssmoking/JPEGImages/smoking_0001.jpg`

* onnx model on streaming video

           `detectnet --model=$NET/ssd-mobilenet.onnx --labels=$NET/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0`

* onnx model on images
           `detectnet --model=$NET/ssd-mobilenet.onnx --labels=$NET/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes "$IMAGES/smoking_*.jpg" $IMAGES/test/smoking_%i.jpg`

## Results
### Image Classification
Below are test images I got from classification project
* [image1](https://imgur.com/o30aPHT)
* [image2](https://imgur.com/bA1Kf3H)
* [image3](https://imgur.com/4vWJw5o)
* [image4](https://imgur.com/DUgNK5L)
* [image5](https://imgur.com/EIepNt8)
* [image6](https://imgur.com/EAXxNmU)
### Image Detection
Unfortuntely detection training terminates unexpectedly, so model can not detect cigarettes. 

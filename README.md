# Smoking Detection
When we walk on the street there are a lot of people smoking cigarettes. Smoking cigarettes are not forbidden if you aren't smoking in a public place. 	Some people smoke in a public area however they are not punished. Because no one sees them while they are smoking. We can solve this problem with AI. If someone holds a cigarette we can find with a camera and punish them for what he/she did.When we walk on the street there are a lot of people smoking cigarettes. Smoking cigarettes are not forbidden if you aren't smoking in a public place. 	Some people smoke in a public area however they are not punished. Because no one sees them while they are smoking. We can solve this problem with AI. If someone holds a cigarette we can find with a camera and punish them for what he/she did.
## The Algorithm
To begin with imageNet for image recognition : [image recognition](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md)

Then I realized the classification is not working in the project so, I started to working in detectNet for object detection
[object detection](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md)

When I did Object Detection, I relized that didn't recognize the cigarettes in the photo, so I labeled the cigarettes from CVAT.
/home/murat/Pictures/annotation_example.png
It took me a long time to label the cigarettes because there were so many cigarette photos. I divided the phptps I tagged into 3 parts: test,validation,training

## Running This Project

At first, I ran jetson-nano in headless mode then I used the [jetson-inference](https://github.com/dusty-nv/jetson-inference) project to find how can I do Image Classification. After that I used Dataset from Kaggle. The Dataset: https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection?resource=download

Below are the steps I have followed to run this applicaiton: 

I converted it to the best model onnx: python3 onnx_export.py --model-dir=models/smoke_nonsmoke/
I tested the model on test images: NET=models/smoke_nonsmoke 

DATASET=data/smoke_nonsmoke

mkdir $DATASET/test_output_smoke $DATASET/test_output_nonsmoke

imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/smoke $DATASET/test_output_smoke

imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/nonsmoke $DATASET/test_output_nonsmoke

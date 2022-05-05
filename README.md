# Smoking Detection
When we walk on the street there are a lot of people smoking cigarettes. Smoking cigarettes are not forbidden if you aren't smoking in a public place. 	Some people smoke in a public area however they are not punished. Because no one sees them while they are smoking. We can solve this problem with AI. If someone holds a cigarette we can find with a camera and punish them for what he/she did.When we walk on the street there are a lot of people smoking cigarettes. Smoking cigarettes are not forbidden if you aren't smoking in a public place. 	Some people smoke in a public area however they are not punished. Because no one sees them while they are smoking. We can solve this problem with AI. If someone holds a cigarette we can find with a camera and punish them for what he/she did.
## The Algorithm
At first, I ran jetson-nano in headless mode then I used the [jetson-inference](https://github.com/dusty-nv/jetson-inference) project to find how can I do Image Classification. After that I used Dataset from Kaggle. The Dataset: https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection?resource=download
I converted it to the best model onnx: python3 onnx_export.py --model-dir=models/smoke_nonsmoke/
I tested the model on test images: NET=models/smoke_nonsmoke 

DATASET=data/smoke_nonsmoke

mkdir $DATASET/test_output_smoke $DATASET/test_output_nonsmoke

imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/smoke $DATASET/test_output_smoke

imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/nonsmoke $DATASET/test_output_nonsmoke

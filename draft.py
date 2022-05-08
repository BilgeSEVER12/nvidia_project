import jetson.inference
import jetson.utils

# to use imagenet app run the following: 
#imagenet --model=models/smoke_nonsmoke/resnet18.onnx --labels=data/labels.txt --input-blob=input_0 --output_blob=output_0 /dev/video0


net = jetson.inference.imageNet('', ['--model=models/smoke_nonsmoke/resnet18.onnx', '--labels=data/labels.txt', '--input-blob=input_0', '--output_blob=output_0' ])

input = jetson.utils.videoSource('/dev/video0')

img = input.Capture()

class_id, confidence = net.Classify(img)

print("Bilge log:", str(class_id), str(confidence))

if(class_id == 0):
  print("Alarm: smoking detected!")
else:
  print("Nothing")

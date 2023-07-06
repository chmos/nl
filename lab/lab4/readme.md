# Lab 4 Resnet
Follow the instruction in [Pytroch Resnet](https://pytorch.org/hub/pytorch_vision_resnet/)

### Generate the list of ImageNet classes
You can use this code to get the list of ImageNet classes
```python
def load_imagenet_classes():
    fp = open('/workpy/labs/ImageNet.txt')
    text = fp.read()
    fp.close()
    
    a = text.split('\n')
    lst = [i.split('\t') for i in a]
    return lst

inet = load_imagenet_classes()
```

### Compare predictions
Show the image and predicted classes (using `inet`). The output looks like this
![prediction](./dog_resnet.png)
<img src="dog_resnet.jpg" alt="prediction" width="200"/>

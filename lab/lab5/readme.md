# Transfer learning and multi-task
[Ref](https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/01-pytorch-object-detection.html)

We will use Resnet-18 as a backbone, add several blocks to form a model that classifies and detects several objects in a picture.

## Data preparation
We will use [Pascal VOC data set](https://pytorch.org/vision/0.8/datasets.html#torchvision.datasets.VOCDetection)

### Find all category names
```python
# get all the names in the Pascal trainset
cname = set()
for i in trainset:
    # i[0] is the image, i[1] is the annotation
    names, border = get_annotation(i[1])
    cname.update(names)
print(cname)

fp = open('/workpy/labs/voc/category.txt', 'w')
for i in cname:
    fp.write(i + '\n')
fp.close()
```

Then you can easily read the names from the text file:
```python
# get all the names in the Pascal trainset
cname = open('/workpy/labs/voc/category.txt').read().split('\n')
print(cname)

print(cname.index('cat'))
```

> ['horse', 'cat', 'car', 'chair', 'motorbike', 'cow', 'bottle', 'pottedplant', 'tvmonitor', 'aeroplane', 'sofa', 'sheep', 'bus', 'boat', 'bird', 'dog', 'train', 'bicycle', 'person', 'diningtable']
> 
> 1

## Transfer learning


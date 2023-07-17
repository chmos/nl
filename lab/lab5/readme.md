# Transfer learning and multi-task
[Ref](https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/01-pytorch-object-detection.html)

We will use Resnet-18 as a backbone, add several blocks to form a model that classifies and detects several objects in a picture.

## Data preparation
We will use [Pascal VOC data set](https://pytorch.org/vision/0.8/datasets.html#torchvision.datasets.VOCDetection)

### load dataset
```python
# Pascal VOC data set
preprocess = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.VOCDetection('/workpy/labs/voc', 
    year = '2010', image_set = 'train', download = True, transform = preprocess)

print('train size =', len(trainset))
```

Note that the images in Pascal VOC have various dimensions. Since Resnet-18 takes input images of $224\times 224$ pixels, we 
resize the image in preprocess.

### data format
You can check the format of a sample:
```python
a = trainset[0]
print(a[0].shape)
print(a[1])
```
> torch.Size([3, 224, 224])
{'annotation': {'folder': 'VOC2010', 'filename': '2008_000008.jpg', 'source': {'database': 'The VOC2008 Database', 'annotation': 'PASCAL VOC2008', 'image': 'flickr'}, 'size': {'width': '500', 'height': '442', 'depth': '3'}, 'segmented': '0', 'object': [{'name': 'horse', 'pose': 'Left', 'truncated': '0', 'occluded': '1', 'bndbox': {'xmin': '53', 'ymin': '87', 'xmax': '471', 'ymax': '420'}, 'difficult': '0'}, {'name': 'person', 'pose': 'Unspecified', 'truncated': '1', 'occluded': '0', 'bndbox': {'xmin': '158', 'ymin': '44', 'xmax': '289', 'ymax': '167'}, 'difficult': '0'}]}}




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


# **Road Segmentation**

### Objective
In the case of the autonomous driving, given an front camera view, the car
needs to know where is the road. In this project, we trained a neural network
to label the pixels of a road in images, by using a method named Fully
Convolutional Network (FCN). In this project, FCN-VGG16 is implemented and trained
with KITTI dataset for road segmentation.



### 1 Code & Files

#### 1.1 My project includes the following files and folders

* [main.py](main.py) is the main code for demos
* [project_tests.py](project_test.py) includes the unittest
* [helper.py](yolo_pipeline.py) includes some helper functions
* [data](data) folder contains the KITTI road data, the VGG model and source images.
* [model](model) folder is used to save the trained model
* [runs](runs) folder contains the segmentation examples of the testing data



#### 1.2 Dependencies & my environment

Miniconda is used for managing my [**dependencies**](env-gpu-py35.yml).

* Python3.5, tensorflow-gpu, CUDA8, Numpy, SciPy
* OS: Ubuntu 16.04


#### 1.3 How to run the code

(1) Download KITTI data (training and testing)

Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)
from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the
dataset in the **data** folder. This will create the folder **data_road** with all
the training a test images.

(2) Load pre-trained VGG

Function ```maybe_download_pretrained_vgg()``` in ```helper.py``` will do
it automatically for you.

(3) Run the code:
```sh
python main.py
```


#### 1.4. Release History
```sh
* 0.1.1
    * Updated documents
    * Date 7 December 2017

* 0.1.0
    * The first proper release
    * Date 6 December 2017
```

---

### 2 Network Architecture

#### 2.1 Fully Convolutional Networks (FCN) in the Wild



FCNs can be described as the above example: a pre-trained model, follow by
1-by-1 convolutions, then followed by transposed convolutions. Also, we
can describe it as **encoder** (a pre-trained model + 1-by-1 convolutions)
and **decoder** (transposed convolutions).

#### 2.2 Fully Convolutional Networks for Semantic Segmentation


The Semantic Segmentation network provided by this
[paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
learns to combine coarse, high layer informaiton with fine, low layer
information. The pooling and prediction
layers are shown as grid that reveal relative spatial coarseness,
while intermediate layers are shown as vertical lines

* The encoder
    * VGG16 model pretrained on ImageNet for classification (see VGG16
    architecutre below) is used in encoder.
    * And the fully-connected layers are replaced by 1-by-1 convolutions.

* The decoder
    * Transposed convolution is used to upsample the input to the
     original image size.
    * Two skip connections are used in the model.

**VGG-16 architecture**

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/03/imagenet_vgg16.png)


#### 2.3 Classification & Loss
we can approach training a FCN just like we would approach training a normal
classification CNN.

In the case of a FCN, the goal is to assign each pixel to the appropriate
class, and cross entropy loss is used as the loss function. We can define
the loss function in tensorflow as following commands.

```sh
logits = tf.reshape(input, (-1, num_classes))
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
```
Then, we have an end-to-end model for semantic segmentation.

### 3 Dataset

![alt text][http://www.cvlibs.net/datasets/kitti/]
In this project, **384** labeled images are used as training data.
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)
from [here](http://www.cvlibs.net/download.php?file=data_road.zip).

There are **4833** testing images are processed with the trained models.
4543 frames from are a video and other 290 images from random places in Karlsruhe.


### 4 Experiments

Some key parameters in training stage, and the traning loss and training
time for each epochs are shown in the following table.

    epochs = 30
    batch_size = 8
    learning_rate = 0.0001



### 5 Discussion

#### 5.1 Good Performance

With only 384 labeled training images, the FCN-VGG16 performs well to find
where is the road in the testing data, and the testing speed is about 6
fps in my laptop. The model performs very well on either highway or urban driving.
Some testing examples are shown as follows:



#### 5.2 Limitations

Based on my test on **4833** testing images. There are two scenarios where
th current model does NOT perform well: (1) turning spot, (2)
over-exposed area.

One possible approach is to use white-balance techniques or image restoration methods
to get the correct image. The other possible approach is to add more
training data with over-exposed scenarios, and let the network to learn
how to segment the road even under the over-expose scenarios.



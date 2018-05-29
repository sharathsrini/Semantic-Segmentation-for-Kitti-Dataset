# **Semantic Segmentation**

### Objective
In the case of the autonomous driving, given an front camera view, the car
needs to know where is the road. In this project, we trained a neural network
to label the pixels of a road in images, by using a method named Fully
Convolutional Network (FCN). In this project, FCN-VGG16 is implemented and trained
with KITTI dataset for road segmentation.
Segmentation is essential for image analysis tasks. Semantic segmentation describes the process of associating each pixel of an image with a class label, (such as flower, person, road, sky, ocean, or car).


![alt text](https://www.mathworks.com/help/vision/ug/semanticsegmentation_transferlearning.png)

##### Applications for semantic segmentation include:

1. Autonomous driving
2. Industrial inspection
3. Classification of terrain visible in satellite imagery
4. Medical imaging analysis

**Semantic segmentation is a natural step in the progression from coarse to fine inference:

The origin could be located at classification, which consists of making a prediction for a whole input.
The next step is localization / detection, which provide not only the classes but also additional information regarding the spatial location of those classes.
Finally, semantic segmentation achieves fine-grained inference by making dense predictions inferring labels for every pixel, so that each pixel is labeled with the class of its enclosing object ore region.

### 1 Code & Files

#### 1.1 Dependencies & my environment

* Python3.5, tensorflow-gpu, CUDA8, Numpy, SciPy
* OS: Ubuntu 16.04


#### 1.2 How to run the code

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


#### 1.3. Release History
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


The pooling and prediction layers are shown as grid that reveal relative spatial coarseness,
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
And thus  we have an end-to-end model for semantic segmentation.

### 3 Dataset


In this project, **384** labeled images are used as training data.
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)
from [here](http://www.cvlibs.net/download.php?file=data_road.zip).

There are **4833** testing images are processed with the trained models.
4543 frames from are a video and other 290 images from random places in Karlsruhe.


### 4 Experiments

Some key parameters in training stage, and the traning loss and training
time for each epochs are shown in the following table.
```sh
    epochs = 30
    batch_size = 8
    learning_rate = 0.0001
```


### 5 Discussion

#### 5.1 Good Performance

With only 384 labeled training images, the FCN-VGG16 performs well to find where is the road in the testing data, and the testing speed is about 6 fps .The model performs very well on either highway or urban driving.Some testing examples are shown as follows:



![alt text](https://github.com/sharathsrini/Semantic-Segmentation-for-Kitti-Dataset/blob/master/runs/1527575647.275648/um_000004.png)
![alt text](https://github.com/sharathsrini/Semantic-Segmentation-for-Kitti-Dataset/blob/master/runs/1527575647.275648/umm_000036.png)
![alt text](https://github.com/sharathsrini/Semantic-Segmentation-for-Kitti-Dataset/blob/master/runs/1527575647.275648/umm_000046.png)
![alt text](https://github.com/sharathsrini/Semantic-Segmentation-for-Kitti-Dataset/blob/master/runs/1527575647.275648/umm_000069.png)



#### 5.2 Limitations

Based on my test on **4833** testing images. There are two scenarios where
th current model does NOT perform well: (1) turning spot, (2)
over-exposed area.One possible approach is to use white-balance techniques or image restoration methods
to get the correct image. The other possible approach is to add more training data with over-exposed scenarios, and let the network to learn how to segment the road even under the over-expose scenarios.



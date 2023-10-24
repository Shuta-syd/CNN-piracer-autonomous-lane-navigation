# Piracer Autonomous Driving via CNN(Conventional Neural Network)
## What is CNN?
The Convolutional Neural Network(CNN) is a Deep Learning algorithm that can extract features from images and be able to understand difference one from the other. The role of CNN is to reduce the images into a form that is easier to process, without losing features that are critical for getting a good prediction. This is important when we are to design an architecture that is not only good at learning features but also scalable to massive datasets.

> recurrent neural networks are commonly used for natural language processing and speech recognition whereas convolutional neural networks (ConvNets or CNNs) are more often utilized for classification and computer vision tasks.

[What are convolutional neural networks? by IBM](https://www.ibm.com/topics/convolutional-neural-networks)  
[Convolutional Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

## How do you build CNN Model?
### Convolution Layer
The objective of the Convolution Operation is to extract the high-level features such as edges, from the input image. CNN need not be limited to only one Convolutional Layer. Conventionally, the first ConvLayer is responsible for capturing the Low-Level features such as edges, color, gradient orientation, etc. With added layers, the architecture adapts to the High-Level features as well, giving us a network that has a wholesome understanding of images in the dataset, similar to how we would.

![Convolution Layer](https://media.tenor.com/qH1i0mBTbgYAAAAC/neural-network.gif)

Image Dimensions = 5 (Height) x 5 (Breadth) x 1 (Number of channels, eg. RGB)

In the above demonstration, the green section resembles our 5x5x1 input image, I. The element involved in the convolution operation in the first part of a Convolutional Layer is called the Kernel represented in color yellow. We have selected Kernel as a 3x3x1 matrix. Most of the time, a 3x3 kernel matrix is very common.

The Kernel shifts 9 times because of Stride Length = 1 (Non-Strided), every time performing Hadamard Product between Kernel and the portion of the image over which the kernel is hovering.

The filter moves to the right with a certain Stride Value till it parses the complete width. Moving on, it hops down to the left of the image with the same Stride Value and repeats the process until the entire image is traversed.
![Convolution operation on a MxNx3 image matrix with a 3x3x3 Kernel](https://cdn-images-1.medium.com/max/1600/1*ciDgQEjViWLnCbmX-EeSrA.gif)

In the case of images with multiple channels (e.g. RGB), the Kernel has the same depth as that of the input image. Matrix Multiplication is performed between Kn and In stack ([K1, I1]; [K2, I2]; [K3, I3]) and all the results are summed with the bias to give us a squashed one-depth channel Convoluted Feature Output.

### Activation Function
Simply put, an activation function is a function that is added into an artificial neural network in order to help the network learn complex patterns in the data. The most important feature in an activation function is its ability to add non-linearity into a neural network.

[Everything you need to know about “Activation Functions” in Deep learning models Vandit Jain](https://towardsdatascience.com/everything-you-need-to-know-about-activation-functions-in-deep-learning-models-84ba9f82c253)

**sigmoid:**
This activation function is here only for historical reasons and never used in real models. It is computationally expensive, causes vanishing gradient problem and not zero-centred. This method is generally used for binary classification problems.

<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1*JHWL_71qml0kP_Imyx4zBg.png&f=1&nofb=1&ipt=36db86707e283e367163dfb2111f06df894451dcbb54739b8915220c750571b0&ipo=images" width="400">

[Don't use sigmoid: Neural Nets](https://kharshit.github.io/blog/2018/04/20/don't-use-sigmoid-neural-nets)  
[Sigmoid and SoftMax Functions in 5 minutes](https://towardsdatascience.com/sigmoid-and-softmax-functions-in-5-minutes-f516c80ea1f9)

**Tanh**  
The output range of the tanh function is (-1, 1) and presents a similar behavior with the sigmoid function. The main difference is the fact that the tanh function pushes the input values to 1 and -1 instead of 1 and 0.

If you compare it to sigmoid, it solves just one problem of being zero-centred. But it still suffers from vanishing gradient problem and is computationally expensive.

<br/>

<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1*1It8846pzYayiC0G_7FIBA.png&f=1&nofb=1&ipt=24769bc4765c28614ff5971c0e152fdba06c3c12c9c8e9d3a35e0da155792a85&ipo=images" width="400">
<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F1.bp.blogspot.com%2F-s7Q1HRNSFoA%2FYOu8seawbSI%2FAAAAAAAAAIc%2FgsM_Iko7PW06RZQhRxcmHuXJJ-4keWY0gCLcBGAsYHQ%2Fs875%2F2.png&f=1&nofb=1&ipt=2a5aaa98f1babbcd2a9dc6b003ced75de59c34bcceea4f9f23428b433526e408&ipo=images" width="400">

<br/>

**ReLU**  
ReLU stands for Rectified Linear Unit. It is the most widely used activation function. It is used in almost all the convolutional neural networks or deep learning.  It is easy to compute and does not saturate and does not cause the Vanishing Gradient Problem. It has just one issue of not being zero centred. It suffers from “dying ReLU” problem.

<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fekababisong.org%2Fassets%2Fieee_ompi%2FReLU.png&f=1&nofb=1&ipt=56b67307ac70ebe48c8e622d6fc13e3f589dd9010c5bb8e08edac3cd7a4fa867&ipo=images" width="400">

<br/>

**Leaky ReLU**  
Leaky ReLU is an improved version of ReLU. It tries to solve the problem of dying ReLU by having a small negative slope in the left side typically 0.01. Leaky ReLu improve the performance fo model by addressing the problem of dying ReLU.

<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fassets-global.website-files.com%2F5d7b77b063a9066d83e1209c%2F60d2474e3a0f7a4010b6129e_pasted%2520image%25200%2520(10).jpg&f=1&nofb=1&ipt=ed3f907f92b9dc11773c5f4fcdbac41f2ccaec8ea04f6833d9d3e7f9ddd0a4a8&ipo=images" width="400">

[Activation functions: ReLU vs. Leaky ReLU](https://medium.com/mlearning-ai/activation-functions-relu-vs-leaky-relu-b8272dc0b1be)

**ELU**  
ELU stands for Exponential Linear Unit. It follows the same concept as ReLU but tries to make the mean activation closer to zero which helps in solving the dying ReLU problem.

<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.5hUNtAxAvoD2MAthTg-rPwHaFj%26pid%3DApi&f=1&ipt=d61f3332f8cf7e4f9b281bd5161f0f670ff9e5f48178f95d79dce02ce9934d9c&ipo=images" width="400">

### Pooling Layer
Similar to the Convolutional Layer, the Pooling layer is responsible for reducing the spatial size of the Convolved Feature. This is to decrease the computational power required to process the data through dimensionality reduction. Furthermore, it is useful for extracting dominant features which are rotational and positional invariant, thus maintaining the process of effectively training the model.

There are two types of Pooling: Max Pooling and Average Pooling. Max Pooling returns the maximum value from the portion of the image covered by the Kernel. On the other hand, Average Pooling returns the average of all the values from the portion of the image covered by the Kernel.
<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftaewanmerepo.github.io%2F2018%2F02%2Fcnn%2Fmaxpulling.png&f=1&nofb=1&ipt=6cc3b2937bea4bb7ad5a38d6d627e121bccd952276dcbfcea2edcd42d9393e1e&ipo=images" width="400"/>

### Fully Connected Layer
The Fully Connected layer is a traditional Multi-Layer Perceptron that uses a softmax activation function in the output layer. The term “Fully Connected” implies that every neuron in the previous layer is connected to every neuron on the next layer.


<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fsds-platform-private.s3-us-east-2.amazonaws.com%2Fuploads%2F74_blog_image_2.png&f=1&nofb=1&ipt=f77c60a042f402c6558ea79d9a535b0c8f863ae49662a436baaf3c1d8d9290ce&ipo=images" width="500"/>

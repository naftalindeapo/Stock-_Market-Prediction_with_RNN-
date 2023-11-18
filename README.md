# Stock Market Prediction with_RNN
This is a simple unified framework for detecting out-of-distribution (OOD) images in neural networks from my [Out-of-Distribution Detection](https://drive.google.com/file/d/1iYIQB629sgECxraShk7qWKXwe9dYhi2e/view?usp=sharing)  research project implemented in [PyThorch](https://pytorch.org). The project explores OOD detection using multiple
techniques, including [MaxSoftmax](https://arxiv.org/abs/1610.02136)), [OpenMax](https://arxiv.org/abs/1511.06233), [Mahalanobis distance](https://arxiv.org/abs/1807.03888), [energy-based methods](https://arxiv.org/abs/2010.03759), and [ODIN](https://arxiv.org/abs/1706.02690), leveraging the pre-trained image classification models [WRN-28-10](https://arxiv.org/abs/1605.07146) and [Dense-BC](https://arxiv.org/abs/1608.06993). Below is an illustration of our OOD detection framework.

![alt text](https://drive.google.com/uc?id=1tcnNRv9HBxI3dsoNXcGTTEckM6W2vLjv)

## Pre-trained Models
In this project, I used four neural networks: (1.) two DenseNet-BC networks trained on Cifar-10 and Cifar-100 respectively, and (2.) Two Wide ResNet networks trained on Cifar-10 and Cifar-100. The PyTorch implementation of the DenseNet-BC and Wide ResNet are provided by [Andreas Veit](https://github.com/andreasveit/densenet-pytorch), and [Sergey Zagoruyko](https://github.com/szagoruyko/wide-residual-networks), respectively. The in-distribution (ID) test error rates of the two models are given in the table below.
|Architecure     | Cifar-10      | Cifar-100 |
| -------------  |:-------------:| ---------:|
| Dense-BC       | 5.16          | 24.06     |
| WRN-28-10      | 5.93          | 25.10     |

## Experimental Results
To evaluate the performance of the OOD detection methods used in our project a range of metrics, including FPR at 95% TPR, detection error, AUROC, AUPR-In, and AUPR-Out were used. The definition of each metric can be found in the paper. The experimental results are shown as follows.

![alt text](https://drive.google.com/uc?id=1pBQbR1xYrz7bAnBlY8GdzKDMfRoDXUtV)

Below is a detailed visualization of the OOD detection performance results using the five methods across the two models.

![alt text](https://drive.google.com/uc?id=1XdWQtQyp3feW_snwUxiFy7t_6oxZDIKp)
![alt text](https://drive.google.com/uc?id=1QJRd69ef4UsNf1euPO0yt8ixhxJoCyzQ) 

Tables 5.2 and 5.3 below show the distribution of scores for the five OOD detection methods on ID datasets
CIFAR-10 and CIFAR-100, versus OOD dataset TinyImageNet(resize), using the two models. The goal is to
understand how well the five methods can distinguish ID and OOD samples. Here we use TinyImageNet(resize)
to demonstrate how the different methods behave; however, similar results can be observed for the other
three OOD datasets.
<div>
    <img src="https://drive.google.com/uc?id=1oZ3X9wYDBn2BPpCtp7Eifz2CrXyw_F2m" style="width: 46%; float: left;" />
    <img src="https://drive.google.com/uc?id=1L93nvK2e7nrTQKX7nnDbystBVbw1aoHQ" style="width: 46%; float: right;" />
 </div>

## Running the code

### Dependencies
- PyTorch
- Anaconda 3
- A GPU-supported device with At least three GPUs

### Downloading OOD datasets
[PyTorch](https://pytorch.org) provides straightforward documentation for loading SVHN OOD dataset found [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html). Below are the links for downloading the remaining three OOD datasets used in our project.
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
- [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

### Download the pre-trained weights
In this project, we used the pre-trained weights on Cifar-10 and Cifar-100 for DenseNet-BC, and Wide ResNet provided in Liang's Git  [repository](https://github.com/facebookresearch/odin). Here are the download links for the four pre-trained models.
- [DenseNet-BC](https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz) trained on Cifar-10
- [DenseNet-BC](https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz) trained on Cifar-100
- [WRN-10-28](https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz) trained on CIFAR-10
- [WRN-10-28](https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet100.pth.tar.gz) trained on CIFAR-100

### License
Refer to the [LICENSE](https://github.com/naftalindeapo/OOD-Detection/blob/main/LICENSE)




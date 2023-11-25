# Stock Market Prediction with_RNN
In this project, we implement three such models: Recurrent Neural Networks (RNN), Long
Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) to predict daily closing prices of AEX
index listed on Euronext Amsterdam. These models were trained using a stock dataset composed of
daily closing prices of 8 indices trading on different stock markets, including AEX index. Three accuracy
tests: MAE, RMSE and MAPE were used to measure the performance of each model. The results have
shown that GRU performed better than RNN and LSTM in terms of predictive accuracy.

This is a simple unified framework for predicting stock prices using machine learning models from my [MSc Mathematical Sciences](https://drive.google.com/file/d/1PD7tn2eRz3VI0Xq71WBdmGFnBG7i5DLP/view?usp=sharing)  research project implemented in [TensorFlow](https://www.tensorflow.org). The project uses three deep neural network models: RNN, LSTM, and GRU to forecast stock time series data composed of closing prices of 8 indices of stocks listed on different stock markets. The selected indices are AEX index, DAXINDX, CAC40,
FTSE100, HNGKNGI, JAPDOWA, NASCOMP, and ATHEX Composite. Below is an illustration of how machine learning can be used in stock market prediction.

![alt text](https://drive.google.com/uc?id=1Pws9qssKrTc_PXQ7F_Q6NSPZjVJZDrVq) 

## RNN architectures
In this project, the following architectures for RNN, LSTM and GRU were used:
<div>
    <img src="https://drive.google.com/uc?id=1VbSAtIs5csGr6Sbj0a5mkyYzLnAJv2Y8" style="width: 46%; float: left;" />
    <img src="https://drive.google.com/uc?id=19HJkyy_Ki7zNj87PIliDnrMvjPsBlmar" style="width: 50%; float: right;" /> 
 </div>
 
## Experimental Results
To evaluate the performance of the OOD detection methods used in our project a range of metrics, including FPR at 95% TPR, detection error, AUROC, AUPR-In, and AUPR-Out were used. The definition of each metric can be found in the paper. The experimental results are shown as follows.
<div>
    <img src="https://drive.google.com/uc?id=1kNP2ZFQTzU-edRO3yXiLOCNB5JUIywtW" style="width: 46%; float: left;" />
    <img src="https://drive.google.com/uc?id=1aoQobTu0wJy_tWFBMpnFE-R8nZbb9klE" style="width: 50%; float: right;" /> 
 </div>

Below is a detailed visualization of the OOD detection performance results using the five methods across the two models.

![alt text](https://drive.google.com/uc?id=1LJQ8xw1JC8VAW6ZdfnE5hgVspA8y2YTw)

![alt text](https://drive.google.com/uc?id=1gOc6LJy2uAPBS3LAA0C9mP6J_B_EMgqZ) 

Tables 5.2 and 5.3 below show the distribution of scores for the five OOD detection methods on ID datasets
CIFAR-10 and CIFAR-100, versus OOD dataset TinyImageNet(resize), using the two models. The goal is to
understand how well the five methods can distinguish ID and OOD samples. Here we use TinyImageNet(resize)
to demonstrate how the different methods behave; however, similar results can be observed for the other
three OOD datasets. 


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




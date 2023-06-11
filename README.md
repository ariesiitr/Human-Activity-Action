# ARIES-Project-
The Aim of Machine Learning Model is to detect various common Activities peformed by humans and classify these activities to their respective classes.
this helps in improving the Robot-Human Interaction and also in Anomaly Detection.
### Link for Pretrained Model- 
https://drive.google.com/drive/folders/1cwBD3hesGUJYjpKmbuGXOClhP7qo9O4u?usp=drive_link

### You can download the dataset For HAR from following link-
https://www.kaggle.com/datasets/pevogam/ucf101/download?datasetVersionNumber=1

## Dataset
UCF-101 dataset consists of 101 classes of various human activities and each class contains several video clips of very short length.
The UCF-101 dataset consists of a total of 13,320 videos, distributed across the 101 action categories.

## Libraries Used
#### Numpy 
#### Pandas
#### Transformers
#### Pytorch
#### cv2
#### Tensorflow
#### Matplotlib

## Model Architecture
The pipeline of the model follows the Base-Deit-16 Model which is connected to Sequential Layer containing: <br>
1. Linear Layer which decreases the number of parameters to work with to 512<br>
2. Relu Activation Function
3. Dropout of 0.3 for Regularization
4. Final Output Layer which outputs the probabilities of input to belong 101-Classes.

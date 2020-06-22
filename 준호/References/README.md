## Intervertebral Disc Segmentation

* Total Case : 997 Case

* Dataset come from Multiple Hospitals, collected by HIRA(Health Insurance Review & Assessment Service)

* Base Network : UNet 2D and Deeplab V3+

* Custom Network : Res-SE-UNet

* Using Keras with backend Tensorflow

* Validation Method : patient 90% used for train (897), 10% used for validation and test(100)

* Project Purpose : Segmentation performance comparison of various algorithms and preprocessing methods.

* Performance Metrics : Dice coefficient score(DSC)

* Methodology 
1) De-identify DICOM files
2) get pixel array from DICOM files
3) split patient to train / test(validation)
4) 897 for train, 100 for test
5) save dataset array to numpy compressed format file(.npz) - as original file
6) resize images to (128, 128), (192, 192), (256, 256) using python, opencv and save to npz file
7) make background label for each target label. (Network should me taught what is background and what is target.)
8) Programming neural network and other things such as custom loss function(dice loss), 
   custom activation function(swish, mish), custom callback object(for saving model and evaluate performance-DSC)
9) start training
10) evaluate to test dataset
11) check performance measurements
12) change parameters, and re-train
12) pruning model weight files 


## References

* A deep learning algorithm may automate intracranial aneurysm detection on MR angiography with high diagnostic performance
  ; Joo, Ahn, Yoon, Bae, Sohn, Lee, "Bae Jun Ho", Park, Choi, Lee

* DSMS-FCN: A Deeply Supervised Multi-scale Fully Convolutional Network for Automatic Segmentation of Intervertebral Disc in 3D MR Images
  ; Zeng, Zheng 

* Fine-Grain Segmentation of the Intervertebral Discs from MR Spine Images Using Deep Convolutional Neural Networks: BSU-Net
  ; Kim, Bae, Masuda, Chung Hwang
  
* IVD-Net: Intervertebral Disc Localization and Segmentation in MRI with a Multi-modal UNet 
  ; Dolz, Desrosiers, Ayed
  
* Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 
  ; Ioffe, Szegedy
  
* Deep Residual Learning for Image Recognition 
  ; He, Zhang, Ren, Sun
  
* Going deeper with convolutions
  ; Szegedy, Liu, Jia, Sermanet, Reed, Anguelov, Erhan, Vanhoucke, Rabinovich
  
* Rethinking Atrous Convolution for Semantic Image Segmentation
  ; Chen, Papandreou, Schroff, Adam
  
* Squeeze-and-Excitation Networks
  ; Hu, Shen, Albanie, Sun, Wu
  
* U-Net: Convolutional Networks for Biomedical Image Segmentation
  ; Ronneberger, Fischer, Brox
  

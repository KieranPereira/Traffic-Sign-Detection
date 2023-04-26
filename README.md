# Traffic-Sign-Detection
This repository contains the code relevant for "Using Object Detection for detection and narration of traffic signs" report.
The code is split into two sections:
1) A Jupyter Notebook containing the majority of code for this project. This notebook is responsible for loading, processing, augmenting and finalising the training, validation and testing dataset. Utalising Google Colab's cloud GPUs, training is also carried out in the notebook, where visuals coded to support points in the report. To use the notebook, a google drive must be mounted containing all initial files of the Mapillary Traffic Sign Dataset in the correct file structure. The notebook relies on access to at least 200GB storage (in a Google Drive) to process the dataset and upload/save training and testing results 
The colab notebook can be found here: https://colab.research.google.com/drive/1v_PefwpB0r91owaywX0E3kEL21JL32sa?usp=sharing
2) The final algorithm as well as model testing Python Scripts found in a separate folder in this repository

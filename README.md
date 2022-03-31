# MBP1413_MultimodalBrainTumorSegmentation
Project repo for class MBP1413 with team members: Phoenix Wilkie, Felix Menze, and Katie Vandeloo. \
 \
This contains code we used and created with some figures and csv files for data analysis. The rest of the outputs and models are found on OneDrive. \
\
Figures_Tables_Splits is the folder that contains all of the figures that were generated, the csv files with data for analysis, and the exact splits of patient cases used for training and testing \
&nbsp; The file structure used for this was: \
&emsp;  MICCAI_BraTS_2018_Data_Training 
```
                  ├── test 
                          ├── HGG 
                              ├── patients 
                              ├── ... 
                              └── patients 
                          └── LGG 
                              ├── patients 
                              ├── ... 
                              └── patients 
                  └── train 
                          ├── HGG 
                              ├── patients 
                              ├── ... 
                              └── patients 
                          └── LGG 
                              ├── patients 
                              ├── ... 
                              └── patients 
```
 \
cnn_model_per_slice is the folder containing all of the models from the cnn per T1w slice \
 \
/BraTs-master/pytorch/checkpoint has all the models from the pre-existing methods, this is found on OneDrive along with ALL of the other files: https://utoronto-my.sharepoint.com/:f:/g/personal/phoenix_wilkie_mail_utoronto_ca/EvzTCODj7N5Pq0QCkYbtrXsBpyYNHJ6uID3Ac9Cr8V8Cjw?e=J9qehZ
```
        BraTs-master 
             |__ contains all the cnn code we made in cnnlearning-master along with some of the analysis files 
             |__ it also contains all of the modified pre-existing model code in the pytorch folder - original code is in old_original_code 
                  |__ output has all of the prediction output from testing for each model contained as images
                  |__ models contains the code used for each model that is run from train_2022 and test_2022 python scripts 
```

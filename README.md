# MBP1413_MultimodalBrainTumorSegmentation
Project repo for class MBP1413 with team members: Phoenix Wilkie, Felix Menze, and Katie Vandeloo. \
 \
Figures_Tables_Splits is the folder that contains all of the figures that were generated, the csv files with data for analysis, and the exact splits of patient cases used for training and testing \
&nbsp; The file structure used for this was: \
&emsp;  MICCAI_BraTS_2018_Data_Training \
```
                  ├── test \
                          ├── HGG \
                              ├── patients \
                              ├── ... \
                              └── patients \
                          └── LGG \
                              ├── patients \
                              ├── ... \
                              └── patients \
                  └── train \
                          ├── HGG \
                              ├── patients \
                              ├── ... \
                              └── patients \
                          └── LGG \
                              ├── patients \
                              ├── ... \
                              └── patients \
```
 \
cnn_model_per_slice is the folder containing all of the models from the cnn per T1w slice \
 \
checkpoint has all the models from the pre-existing methods, this is found on OneDrive: \
```
        BraTs-master \
             |__ contains all the cnn code we made in cnnlearning-master along with some of the analysis files \
             |__ it also contains all of the modified pre-existing model code in the pytorch folder - original code is in old_original_code \
                  |__ output has all of the prediction output from testing for each model contained as images \
                  |__ models contains the code used for each model that is run from train_2022 and test_2022 python scripts \
```

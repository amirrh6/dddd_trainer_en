- [x] README.md is translated to English via [Google Translate](https://github-com.translate.goog/sml2h3/dddd_trainer?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en) and it ain't smooth so any sort of help from Chinese speakers is apreciated.
- - [x] Removed `-i https://pypi.douban.com/simple` from `pip install -r requirements.txt` command. 
- [x] 2 Test Datasets are uploaded to Google Drive instead of Lanzoum.
- [x] Added `onnx` to requirements.txt file.


# dddd_trainer OCR training tool for my younger brother

### The training tool used by my brother OCR is officially open source today! [ddddocr](https://github.com/sml2h3/ddddocr)

### The project only supports N card training. Don’t look at A card or other cards yet.

### The project is developed based on Pytorch and supports cnn and crnn training, breakpoint recovery, automatic export of onnx models, and also supports seamless deployment using [ddddocr](https://github.com/sml2h3/ddddocr) and [ocr_api_server](https://gitee.com/fkgeek/ocr_api_server) .

### Training environment support

Windows/Linux

Macos only supports CPU training

## 1. Necessary environment configuration for deep learning (not only required by this project, but required by all deep learning projects, except CPU training)

### Before starting this tutorial, please go to [the pytorch](https://pytorch.org/get-started/locally/) official website to check the pytorch version supported by your system and hardware. Note that for N cards before the 30 series, such as 2080Ti, please choose the version below cuda11 (for example: CUDA 10.2). If it is a 30 series N card, Only CUDA 11 version is supported. Please select CUDA 11 or above version (for example: CUDA 11.3), and then complete the pytorch installation according to the pytorch installation command displayed according to the selected conditions. Due to the version update speed of pytorch, many pypi sources only cache the cpu version, CUDA The version needs to be installed on the official website.

### Install CUDA and CUDNN

Choose according to your graphics card model and system

[cuda](https://developer.nvidia.com/cuda-downloads)

[cudnn](https://developer.nvidia.com/zh-cn/cudnn)

Note that the cuda version number supported by cudnn should correspond to the cuda version number you installed. Different versions of cuda support different graphics cards. <b> For the 20 series brainless version, choose version 10.2 cuda, and for the 30 series brainless version, choose version 11.3 cuda</b > . If there is any problem here, please Baidu, it's a basic question.

## 2. Training part

- All the following variables are replaced by {param} format, which means that they can be modified according to your own needs, and you do not need to bring {} when using it. For example, if you create a new training project in the step, you can write it directly when using it.

`python app.py create test_project`

- ### 1. Clone this project locally

~~`git clone https://github.com/sml2h3/dddd_trainer.git`~~

`git clone https://github.com/amirrh6/dddd_trainer_en.git`

- ### 2. Enter the project directory and install the dependencies required for this project

~~`pip install -r requirements.txt -i https://pypi.douban.com/simple`~~

`pip install -r requirements.txt`

- ### 3. Create a new training program

`python app.py create {project_name}`

If you want to create a CNN project, you can add the --single parameter. The CNN project identifies, for example, what category the picture is. For example, if there is only one word on the picture, identify what word the picture is (there are multiple pictures on the picture). Do not use CNN mode for text), and for example, it is more appropriate to use CNN mode to distinguish whether a lion or a rabbit is in a picture. Please do not use --single for most OCR needs.

`python app.py create {project_name} --single`

project_name is the project name, try not to name it with special symbols

- ### 4. Prepare data

    Project supports two forms of data
    
    ### A. Import from file name
        
    The pictures are all in the same folder and named similarly. /root/images_set is the directory where the pictures are located, which can be any directory address.

    ```
  /root/images_set/
    |---- abcde_randomHashValue.jpg
    |---- sdae_randomHashValue.jpg
    |---- 酱闷肘子_randomHashValue.jpg
  
  ```
    
    As shown below

    ![image](https://cdn.wenanzhe.com/img/mkGu_000001d00f140741741ed9916240d8d5.jpg)

    Then the picture naming can be

    `mkGu_000001d00f140741741ed9916240d8d5.jpg`

    ### In order to consider various situations, dddd_trainer will not automatically handle the case problem. If you want to train the case, you need to mark the case yourself when labeling the sample, as in the above example

    ### B. Import from file

    Limited by possible sample organization forms or special characters, this project supports importing data from txt documents. The data set directory must contain `labels.txt` files and `images` folders, where /root/images_set is the directory where the images are located, and can be any directory address.
    
`labels.txt` The file contains the relative paths of all images `/root/images_set/images` under the directory , and there can be directories under it. `/root/images_set/images` `/root/images_set/images`

#### Of course, in this mode, the file name of the image is arbitrary, and it may or may not have a specific label, because we do not get the label of the image from here.

As follows

- The form without directory under a.images

    ```
  /root/images_set/
    |---- labels.txt
    |---- images
          |---- randomhashValue.jpg
          |---- randomhashValue.jpg
          |---- 酱闷肘子_randomhashValue.jpg
  
  labels.txt The file content is（The \ttab character is the separator between the file name and label of each line.）
  randomhashValue.jpg\tabcd
  randomhashValue.jpg\tsdae
  酱闷肘子_randomhashValue.jpg\t酱闷肘子
  ```
  b.In the form of a directory under images
    ```
  /root/images_set/
    |---- labels.txt
    |---- images
          |---- aaaa
                |---- randomhashValue.jpg
          |---- 酱闷肘子_randomhashValue.jpg
  
  The content of the labels.txt file is（The \ttab character is the separator between the file name and label of each line）
  aaaa/randomhashValue.jpg\tabcd
  aaaa/randomhashValue.jpg\tsdae
  酱闷肘子_randomhashValue.jpg\t酱闷肘子
  
  ```
  
  ### In order for novices to better understand the content of this part, this project also provides two sets of basic data sets for testing.

    [Data set one](https://drive.google.com/file/d/1pWCQC3TV42zAfJc_338_G1Lt1Yyg7L0w/view?usp=sharing)
    [Data set two ](https://drive.google.com/file/d/1TJdM1YaGzylSWuA2Nd4Txi6LrwuqVHdt/view?usp=sharing)
- ### 5. Modify configuration file
```yaml
Model:
    CharSet: []     # Don’t touch the character set, it will be generated automatically.
    ImageChannel: 1 # The number of image channels. If you want to train with grayscale images, set it to 1, and if you want to use color images for training, set it to 3. If set to 1, the data set is a color image, and the project will automatically convert the read color image into a grayscale image in the memory during the training process. There is no need to modify it in advance and this setting will not modify the local image.
    ImageHeight: 64 # The height of the image after automatic scaling. The unit is px. The height must be a multiple of 16. The image will be automatically scaled.
    ImageWidth: -1  # The width of the image after automatic scaling, in px. If this item is set to -1, it will be automatically adjusted according to the situation.
    Word: false     # Whether it is a CNN model or not is controlled by parameters when creating the project. Do not modify it yourself.
System:
    Allow_Ext: [jpg, jpeg, png, bmp]  # Supported image suffixes, unsatisfied images will be automatically ignored
    GPU: true                         # Whether to enable GPU for training. To use GPU training, you need to refer to step 1 to install the environment.
    GPU_ID: 0                         # GPU device number, 0 is the first graphics card
    Path: ''                          # The root directory of the data set will be automatically generated during the image caching step. You do not need to change it unless the data set address is changed.
    Project: test                     # Project name is {project_name}
    Val: 0.03                         # The data volume ratio of the verification set, 0.03 is 3%. When caching the data, 3% of the images will be automatically selected for data verification during the training process. After modifying this value, the data needs to be cached again.
Train:
    BATCH_SIZE: 32                                    # The size of each batch_size during training mainly depends on your video memory or memory size. You can test more according to your own situation, usually a multiple of 16, such as 16, 32, 64, 128
    CNN: {NAME: ddddocr}                              # Feature extraction model, currently supported values ​​are ddddocr, effnetv2_l, effnetv2_m, effnetv2_xl, effnetv2_s, mobilenetv2, mobilenetv3_s, mobilenetv3_l
    DROPOUT: 0.3                                      # Non-professionals should not move
    LR: 0.01                                          # Initial learning rate
    OPTIMIZER: SGD                                    # Optimizer, don't move
    SAVE_CHECKPOINTS_STEP: 2000                       # Save the model every how many steps
    TARGET: {Accuracy: 0.97, Cost: 0.05, Epoch: 20}   # The goal of the end of training, when both are met, the training will automatically end and the onnx model will be saved. Accuracy is the minimum accuracy that needs to be met, Cost is the minimum loss that needs to be met, and Epoch is the minimum number of training rounds that needs to be met.
    TEST_BATCH_SIZE: 32                               # The size of each batch_size during testing mainly depends on your video memory or memory size. You can test more according to your own situation, usually a multiple of 16, such as 16, 32, 64, 128
    TEST_STEP: 1000                                   # Test every few steps


```
The configuration file is located in the root directory of this project `projects/{project_name}/config.yaml`

- ### 6. Caching data

`python app.py cache {project_name} /root/images_set/`

If you are reading data from labels.txt

`python app.py cache {project_name} /root/images_set/ file`

- ### 7. Start training or resume training

`python app.py train {project_name}`

- ### 8. Deployment

`You guys train first, I will adapt ddddocr and ocr_api_server. After the adaptation, I will continue to update the document`

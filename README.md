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
    CharSet: []     # 字符集，不要动，会自动生成
    ImageChannel: 1 # 图片通道数，如果你想以灰度图进行训练，则设置为1，彩图，则设置为3。如果设置为1，数据集是彩图，项目会在训练的过程中自动在内存中将读取到的彩图转为灰度图，并不需要提前自己修改并且该设置不会修改本地图片
    ImageHeight: 64 # 图片自动缩放后的高度，单位为px,高度必须为16的倍数，会自动缩放图像
    ImageWidth: -1  # 图片自动缩放后的宽度，单位为px，本项若设置为-1，将自动根据情况调整
    Word: false     # 是否为CNN模型，这里在创建项目的时候通过参数控制，不要自己修改
System:
    Allow_Ext: [jpg, jpeg, png, bmp]  # 支持的图片后缀，不满足的图片将会被自动忽略
    GPU: true                         # 是否启用GPU去训练，使用GPU训练需要参考步骤一安装好环境
    GPU_ID: 0                         # GPU设备号，0为第一张显卡
    Path: ''                          # 数据集根目录，在缓存图片步骤会自动生成，不需要自己改，除非数据集地址改了
    Project: test                     # 项目名称 也就是{project_name}
    Val: 0.03                         # 验证集的数据量比例，0.03就是3%，在缓存数据时，会自动选则3%的图片用作训练过程中的数据验证，修改本值之后需要重新缓存数据
Train:
    BATCH_SIZE: 32                                    # 训练时每一个batch_size的大小，主要取决于你的显存或内存大小，可以根据自己的情况，多测试，一般为16的倍数,如16，32，64，128
    CNN: {NAME: ddddocr}                              # 特征提取的模型，目前支持的值为ddddocr,effnetv2_l,effnetv2_m,effnetv2_xl,effnetv2_s,mobilenetv2,mobilenetv3_s,mobilenetv3_l
    DROPOUT: 0.3                                      # 非专业人员不要动
    LR: 0.01                                          # 初始学习率
    OPTIMIZER: SGD                                    # 优化器，不要动
    SAVE_CHECKPOINTS_STEP: 2000                       # 每多少step保存一次模型
    TARGET: {Accuracy: 0.97, Cost: 0.05, Epoch: 20}   # 训练结束的目标，同时满足时自动结束训练并保存onnx模型，Accuracy为需要满足的最小准确率，Cost为需要满足的最小损失，Epoch为需要满足的最小训练轮数
    TEST_BATCH_SIZE: 32                               # 测试时每一个batch_size的大小，主要取决于你的显存或内存大小，可以根据自己的情况，多测试，一般为16的倍数,如16，32，64，128
    TEST_STEP: 1000                                   # 每多少step进行一次测试


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

# ResNetFCN
A tf2 implementation of a FCN (using a ResNet for transfer learning) over the VOC 2012 dataset.

# Steps to Use
1. Run DataPrep.ipynb (Downloads VOC 2012 dataset and makes tf.Record dataset).
2. Run ModelPrep.ipynb (Declares ResNetFCN, trains model, and converts to tf.lite model).
3. Run Inference.ipynb (Infer test image with ResNetFCN, outputs test image with labeled/colored mask).

# Train model not in notebook
If you need to train the model not in a jupyter notebook (for example, on a remote server),
the model can be trained by running train_model.py.

# Run model in docker container 
Within the folder ResNetFCN_app, there is a set of files to run the model in a docker container.
1. Build docker container: docker build -t resnetfcn_app .
2. Run docker container (replace <test_img.jpg> with desired test img file name): docker run -e IMG_PATH='<test_img.jpg>' -v $(pwd)/output:/usr/src/app/output -v $(pwd)/<test_img.jpg>:/usr/src/app/<test_img.jpg> resnetfcn_app
3. Results are placed into the folder called output. 

# File structure
File structure used for project.
``` bash
.
├── DataPrep.ipynb
├── Inference.ipynb
├── ModelPrep.ipynb
├── ResNetFCN_App
│   ├── Dockerfile
│   ├── output
│   │   └── pred_img_mask.jpg
│   ├── requirements.txt
│   ├── resnetfcn_app.py
│   ├── resnetfcn_lite.tflite
│   └── test_pic.jpg
├── ResNetFCN_VOC2012.h5
├── VOCdevkit.zip
├── f16_lite_model
│   └── ResNetFCN_VOC2012_f16.tflite
├── img_mask.jpg
├── pred_img_mask.jpg
├── sample_pred.png
├── test_pic.jpg
├── tfData
│   ├── train_record.tfrecords
│   └── val_record.tfrecords
└── train_model.py
```

# Sample prediction
![alt text](sample_pred.png)

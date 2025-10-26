## **MVPFusion: Multi-View Perception Network for Multi-Focus Image Fusion**

### Submission to the journal Neural Networks

This repository contains the implementation of MVPFusion, a multi-view perception network for multi-focus image fusion. To evaluate the effectiveness of our method, we compared it with other state-of-the-art methods on three multi-focus image datasets,achieving state-of-the-art results in both subjective visual effects and objective measurement outcomes, while also demonstrating significant advantages in lightweight design and time efficiency. The model weights will be uploaded after the paper is accepted.

### Document

<pre>MVPFusion/
├── Datasets/
│   ├── Eval/
│          └── Lytro
│                 ├── sourceA/
│                        ├── lytro-01-A.jpg
│                        ├── lytro-02-A.jpg
│                        ├── ...
│                        ├── lytro-20-A.jpg
│                 └── sourceB/
│                        ├── lytro-01-B.jpg
│                        ├── lytro-02-B.jpg
│                        ├── ...    
│                        ├── lytro-20-B.jpg
│   └── Train&Valid/
│          └── DUTS_MFF/
│                 ├── train/
│                        ├── decisionmap/
│                        ├── sourceA/
│                        └── sourceB/
│                 └── validate
│                        ├── decisionmap/
│                        ├── sourceA/
│                        └── sourceB/
├── Nets/
│   ├── MVPFusion.py
│   └── SwinTransformer.py
├── RunTimeData/
│   └── Model weights
│          └── best_network.pth
├── Utilities/
├── Fusion.py
├── Training.py
├── requirements.txt
└── README.md</pre>


| File name        | Explanation                                                  |
| ---------------- | ------------------------------------------------------------ |
| Datasets         | Training dataset and the testing dataset (There are no restrictions on image naming, they just need to correspond one-to-one in the folder) |
| Nets             | Network structure                                            |
| Results          | Test output                                                  |
| RunTimeData      | Trained model weights                                        |
| Utilities        | Utility function                                             |
| Fusion.py        | For testing                                                  |
| Training.py      | For training                                                 |
| requirements.txt | Dependencies required by the model                           |

### Preparation

#### Dependencies

Please refer to requirements.txt for details.

#### Data Preparation

[DUTS](https://paperswithcode.com/dataset/duts)(training dataset)

[Lytro](https://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset)(testing dataset)

[MFFW](https://github.com/lmn-ning/ImageFusion/tree/main/FusionDiff/Dataset/Multi-Focus-Images/valid)(testing dataset)

[Grayscale](https://github.com/yuliu316316/MFIF/tree/master/sourceimages/grayscale)(testing dataset)

### Training

For training, please run:

`python Training.py`

### Testing

For testing, please run:

`python Fusion.py`

### Results

The output results will be stored in ./Results

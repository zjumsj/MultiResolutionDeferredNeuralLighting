# A Multi-Resolution Network Architecture for Deferred Neural Lighting

Tensorflow implementation of paper *A Multi-Resolution Network Architecture for Deferred Neural Lighting* which is comming soon in CASA 2022.

![teaser](./fig/total2_casa.png)
[Supplemantary video](https://zjumsj.github.io/MRNDNL.github.io/)

Our method considerably improves the high-frequency details as well as temporal stability in rendering animation in neural relighting. The code is partially based on [DeferredNeuralLighting](https://github.com/msraig/DeferredNeuralLighting) and we refer to their method as *baseline*.   



## Setup

The code is tested on python=3.7, dependencies:

```
tensorflow-gpu==1.15
tensorboard==1.15
numpy
tqdm
configargparse
opencv-python
OpenEXR==1.3.2
```

## Dataset  

We provide two training dataset (Pixiu statuette, Cluttered scene). [Download](TODO)

The dataset is derived from [DeferredNeuralLighting](https://github.com/msraig/DeferredNeuralLighting) and modified. Note that we use only one network for each scene, instead of 13 ones to cover different viewing directions, as we find it enough for good results. We integrate the dataset from 13 clusters into one and remove the duplicate. To further test the generalization capacity and temporal consistency of the model, we create the reduced dataset from the full one by random selection. The scale of each dataset is shown below. 

| | full | reduced  |
|----|----|----|
|Cluttered scene| 16,521 | 2,000 |
|Cat| 6,384 | 1,000 |
|Ornamental fish| 13,032 | 2,000 |
|Pixiu statuette| 13,597 | 2,000 |
|Sword| 10,608 | 2,000 |

## Test 
### Pretrained models

We provide pre-trained models of all the five scenes including baseline models. We also provide models trained on both full and reduced dataset.

The pretrained models can be found in folder *models* from [Google Drive](https://drive.google.com/drive/folders/1h2F9OFWf814opyvX_XiIc3OBJdyv9mEw?usp=sharing) or [OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3140103086_zju_edu_cn/EoRrq2-SRjlLt7t4HLsXkwABYLHcJMetyH-hpYr3g84aeA?e=8kvKoj).

The models should be unzipped and put in `./models`. You should get something like this:  
```
.
└── models
      ├── furscene    					
      │      ├── baseline_furscene_full   		
      │      ├── baseline_furscene_s2000   		
      │      ├── multires_furscene_full
      │      └── multires_furscene_s2000
```

### Test dataset

The test dataset can be found in folder *test* from [Google Drive](https://drive.google.com/drive/folders/1h2F9OFWf814opyvX_XiIc3OBJdyv9mEw?usp=sharing) or [OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3140103086_zju_edu_cn/EoRrq2-SRjlLt7t4HLsXkwABYLHcJMetyH-hpYr3g84aeA?e=8kvKoj).

The dataset should be unzipped and put in `./test`. You should get something like this:  
```
.
└── test
     ├── furscene    					
     │      ├── RotLight_v1   		
     │      └── RotView_v2
```

### Run rendering

The pretrainded models and test dataset can be used to reproduce comparisons shown in our supplementary video. Before running the test script, make sure you have downloaded and put everything in the correct places.

Run
```
bash test_furscene.sh
bash test_cat.sh
bash test_fish.sh
bash test_pixiu.sh
bash test_sword.sh
```

Then, you can run `python gen_video_from_seq.py val` to generate videos from image sequences. The outputs can be found in `./val`.    

## Train

To train a model, you can download a training dataset [here](TODO) and put it in `./dataset`. You should get something like this:

```
.
└── dataset
       ├── furscene    					
       │      ├── fulldataset   		
       │      └── reduced_2000
       └── pixiu
              ├── fulldataset   		
              └── reduced_2000
```

Here is an example of training on full dataset of Cluttered scene.
```
python train_multires.py --config config/config_multires_furscene.txt --device 0
```

Train baseline model
```
python train_baseline.py --config config/config_baseline_furscene.txt --device 0
```

### More

Train with full multi-scale loss
```
python train_multires.py --config config/config_fullLoss.txt --device 0
```

Train with progressive strategy
```
bash train_progressive.sh
```


## Citation


<!-- TODO: may modify citation when paper is officially published-->

```
@article{ma2022multi,
  title={A Multi-Resolution Network Architecture for Deferred Neural Lighting},
  author={Ma, Shengjie and Wu, Hongzhi and Ren, Zhong and Zhou, Kun},
  year={2022},
  booktitle={CASA},
}
```

If you use the dataset, please also cite

```
@article{gao2020deferred,
  title={Deferred neural lighting: free-viewpoint relighting from unstructured photographs},
  author={Gao, Duan and Chen, Guojun and Dong, Yue and Peers, Pieter and Xu, Kun and Tong, Xin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={39},
  number={6},
  pages={258},
  year={2020},
  month={December},
  publisher={ACM}
}
```



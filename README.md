# Decoupled attention network for text recognition

This is a pytorch-based implementation for paper [Decoupled attention network for text recognition](https://arxiv.org/abs/1912.10205) (AAAI-2020).


## Requirements

We recommend you to use [Anaconda](https://www.anaconda.com/) to manage your libraries.

- [Python 2.7](https://www.python.org/) (The data augmentation toolkit does not support python3)
- [PyTorch](https://pytorch.org/) (We have tested 0.4.1 and 1.1.0)
- [TorchVision](https://pypi.org/project/torchvision/)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/#)
- [Colour](https://pypi.org/project/colour/)
- [LMDB](https://pypi.org/project/lmdb/)
- [editdistance](https://pypi.org/project/editdistance/)

Or use [pip](https://pypi.org/project/pip/) to install the libraries. (Maybe the torch is different from the anaconda version. Please check carefully and fix the warnings in training stage if necessary.)

```bash
    pip install -r requirements.txt
```
Besides, a [data augmentation toolkit](https://github.com/Canjie-Luo/Scene-Text-Image-Transformer) is used for handwritten text recognition.

## Updates
Nov 28, 2020

Thanks to huizhang0110, we find a [bug](https://github.com/Wang-Tianwei/Decoupled-attention-network/issues/34) which results in higher performance on IAM dataset.

The result on IAM dataset should be corrected as (CER 7.0, WER 20.6). 

Conclusions are not affected.


Dec 30, 2019

Trained models:

[Google Drive](https://drive.google.com/drive/folders/1MK0WUH-ofIPT4ZNTbcb0sburatJyEF1X?usp=sharing)

[Baidu Netdisk](https://pan.baidu.com/s/1XUdYI6KoLnUbCAmM1JRBNw) password: sds8

The handwritten models are well trained (IAM-CER 6.4, IAM-WER 19.6). The scene models are single-directional and nearly well trained (IIIT5K 93.3).

## Data Preparation
### Offline handwritten text
Here we provide the codes for IAM dataset. For RIMES, please prepare it by yourself.

IAM database can be downloaded from [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database). 

For convenience, we provide the processed annotations (in `data/IAM/`) and the dataloader (`dataset_hw.py`). You only need to download [data/lines](http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines) and [data/words](http://www.fki.inf.unibe.ch/DBs/iamDB/data/words) then put the unzipped files into the folder `data/IAM/`.

Note that all data is loaded into memory at once. Make sure there is enough memory (about 5 GB).

### Scene text
Please convert your own dataset to **LMDB** format by using the [tool](https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py) (run in **Python 2.7**) provided by [@Baoguang Shi](https://github.com/bgshih). 

You can also download the training ([NIPS 2014](http://www.robots.ox.ac.uk/~vgg/data/text/), [CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)) and testing datasets prepared by us. 

- [BaiduCloud (about 20G training datasets and testing datasets in **LMDB** format)](https://pan.baidu.com/s/1TqZfvoEhyv57yf4YBjSzFg), password: l8em
- [Google Drive (testing datasets in **LMDB** format)](https://drive.google.com/open?id=1NAs78a38xkl1MhodoD7BM0Lh3v_sFwYs)
- [OneDrive (testing datasets in **LMDB** format)](https://1drv.ms/f/s!Am3wqyDHs7r0hkHUYy0edaC2UC3c)

The raw pictures of testing datasets can be found [here](https://github.com/chengzhanzhan/STR).

## Training and Testing

Modify the path in configuration files (`cfgs_scene.py` for scene, `cfgs_hw.py` for handwritten). Make sure the import is correct in `line 12, main.py`. Then:

```bash
	python main.py
```

## Citation

```
@InProceedings{DAN_aaai20,
  author = {Tianwei Wang and Yuanzhi Zhu and Lianwen Jin and Canjie Luo and Xiaoxue Chen and Yaqiang Wu and Qianying Wang and Mingxiang Cai}, 
  title = {Decoupled attention network for text recognition}, 
  booktitle ={AAAI Conference on Artificial Intelligence}, 
  year = {2020}
}
```

## Attention
The project is only free for academic research purposes.

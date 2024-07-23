# Outlier-Aware Test-Time Adaptation with Stable Memory Replay
[[paper](https://arxiv.org/abs/2407.15773)]
## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yaml
conda activate Benchmark_TTA 
```

## Structure of Project

This project is based on a [TTA-Benchmark](https://github.com/yuyongcan/Benchmark-TTA) containing several directories. Their roles are listed as follows:

+ ./cfgs: the config files for each dataset and algorithm are saved here.
+ ./robustbench: an official library we use to load robust datasets and models. 
+ ./src/
  + data: we load our datasets and dataloaders by code under this directory.
  + methods: the code for the implementation of various TTA methods.
  + models: the various models' loading process and definition rely on the code here.
  + utils: some useful tools for our projects. 

## Run

This repository allows to study a wide range of different datasets, models, settings, and methods. A quick overview is given below:

- **Datasets**
  
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
  
  - `LSUN-C` [LSUN](https://github.com/fyu/lsun)
  
  - `SVHN-C` [SVHN](http://ufldl.stanford.edu/housenumbers/)
  
  - `Tiny-ImageNet-C` [Tiny-ImageNet-C](https://zenodo.org/record/2536630)
  
  - `Textures-C` [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
  
  - `Places365-C` [Places365](http://places2.csail.mit.edu/) 

The dataset directory structure is as follows:

  
  	|-- datasets 
  	
  	        |-- cifar-10
  	
  	        |-- cifar-100
  	
  	        |-- ImageNet
  	
  	                |-- train
  	
  	                |-- val
  	
  	        |-- ImageNet-C
  	
  	        |-- CIFAR-10-C
  	
  	        |-- CIFAR-100-C
        
            |-- LSUN_resize-C
      
            |-- PLACES365-C

            |-- SVHN-C

            |-- Textures-C

            |-- Tiny-ImageNet-C

**For OOD datasets**, you can generate the corrupted datasets according to the instructions in this [repository](https://github.com/yuyongcan/generating_outlier) or [robustbench](https://github.com/hendrycks/robustness).

- **Models**
  
  - You can train the source model by script in the ./pretrain directory.
  
  - You can also download our checkpoint from [here](https://drive.google.com/drive/folders/1QQUqG4Kqw9TC-1FBX7mOak7iU488_G0w?usp=drive_link).

- **Methods**
  - The repository currently supports the following methods: source, [PredBN](https://arxiv.org/abs/2006.10963), [TENT](https://openreview.net/pdf?id=uXl3bZLkr3c),
    [EATA](https://arxiv.org/abs/2204.02610), [RoTTA](https://arxiv.org/abs/2303.13899), [SoTTA](https://arxiv.org/abs/2310.10074), [OWTTT](https://arxiv.org/abs/2308.09942),
    [CoTTA](https://arxiv.org/abs/2203.13591), [SAR](https://openreview.net/forum?id=g2YraF75Tj).


- **Modular Design**
  - Adding new methods should be rather simple, thanks to the modular design.

### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.

Next, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`.

download the checkpoints of pre-trained models from [here](https://drive.google.com/drive/folders/1QQUqG4Kqw9TC-1FBX7mOak7iU488_G0w?usp=drive_link) and put it in ./ckpt
#### How to reproduce
The entry file for algorithms is **test-time-eva-baseline.sh**

To evaluate these methods, modify the DATASET and METHOD in test-time-eva.sh

and then

```shell
bash test-time-eva-baseline.sh
```

## Acknowledgements

+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ SAR [official](https://github.com/mr-eggplant/SAR)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ SoTTA [official](https://github.com/taeckyung/SoTTA)
+ OWTTT [official](https://github.com/Yushu-Li/OWTTT)
+ RoTTA [official](https://github.com/BIT-DA/RoTTA)


# DPI
## 1. Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). Also, please make sure you have at least one NVIDIA GPU with Linux x86_64 Driver Version >= 510.47 (compatible with CUDA 11.6). We applied distributed training on 4 NVIDIA RTX A6000 with 48 GB graphic memory, and the batch size corresponds to it. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.

#### 1.1 Create and activate a new virtual environment

```
conda env create -f ~/miniconda3/envs/dpi_environment.yml
conda activate dpi
```



#### 1.2 Install the package and other requirements

(Required)

```
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch

git clone https://github.com/cong-003/DPI
cd DPI
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```


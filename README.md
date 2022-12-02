# DPI
## 1. Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). Also, please make sure you have at least one NVIDIA GPU with Linux x86_64 Driver Version >= 510.47 (compatible with CUDA 11.6 or below). We applied distributed training on 4 NVIDIA RTX A6000 with 48 GB graphic memory, and the batch size corresponds to it. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.

#### 1.1 Create and activate a new virtual environment

```
conda env create -n dpi python=3.7
conda activate dpi
```



#### 1.2 Install the package and other requirements

(Required)

```
pip3 install numpy==1.21.0 pandas==1.3.5 matplotlib==3.2.2 scikit-learn==1.0.2 jupyter==1.0.0 tqdm==4.63.0 GPUtil==1.4.0
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

git clone https://github.com/cong-003/DPI
cd DPI
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```


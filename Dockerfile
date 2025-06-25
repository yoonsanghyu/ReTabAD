# Base Image (PyTorch2.3.1, CUDA 11.8)
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# set timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    htop \
    python3-pip \
    python3-dev \
    cmake \
    llvm \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libfreetype6-dev \
    libxft-dev \
    net-tools \ 
    openssh-server \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install Python Dependencies from requirements_anollm.txt
RUN pip install \
accelerate==1.7.0 \
addict==2.4.0 \
aiohappyeyeballs==2.6.1 \
aiohttp==3.12.13 \
aiosignal==1.3.2 \
async-timeout==5.0.1 \
attrs==25.3.0 \
certifi==2025.6.15 \
charset-normalizer==3.4.2 \
contourpy==1.3.2 \
cycler==0.12.1 \
datasets==3.6.0 \
dill==0.3.8 \
dotenv==0.9.9 \
filelock==3.18.0 \
fonttools==4.58.4 \
frozenlist==1.7.0 \
fsspec==2025.3.0 \
hf-xet==1.1.4 \
huggingface-hub==0.33.0 \
idna==3.10 \
Jinja2==3.1.6 \
joblib==1.5.1 \
kiwisolver==1.4.8 \
llvmlite==0.44.0 \
MarkupSafe==3.0.2 \
matplotlib==3.10.3 \
mpmath==1.3.0 \
multidict==6.5.0 \
multiprocess==0.70.16 \
networkx==3.4.2 \
numba==0.61.2 \
numpy==2.2.6 \
nvidia-cublas-cu11==11.10.3.66 \
nvidia-cublas-cu12==12.1.3.1 \
nvidia-cuda-cupti-cu12==12.1.105 \
nvidia-cuda-nvrtc-cu11==11.7.99 \
nvidia-cuda-nvrtc-cu12==12.1.105 \
nvidia-cuda-runtime-cu11==11.7.99 \
nvidia-cuda-runtime-cu12==12.1.105 \
nvidia-cudnn-cu11==8.5.0.96 \
nvidia-cudnn-cu12==8.9.2.26 \
nvidia-cufft-cu12==11.0.2.54 \
nvidia-curand-cu12==10.3.2.106 \
nvidia-cusolver-cu12==11.4.5.107 \
nvidia-cusparse-cu12==12.1.0.106 \
nvidia-nccl-cu12==2.20.5 \
nvidia-nvjitlink-cu12==12.9.86 \
nvidia-nvtx-cu12==12.1.105 \
packaging==25.0 \
pandas==2.3.0 \
peft==0.11.1 \
pillow==11.2.1 \
propcache==0.3.2 \
psutil==7.0.0 \
pyarrow==20.0.0 \
pyod==2.0.1 \
pyparsing==3.2.3 \
python-dateutil==2.9.0.post0 \
python-dotenv==1.1.0 \
pytz==2025.2 \
PyYAML==6.0.2 \
regex==2024.11.6 \
requests==2.32.4 \
safetensors==0.5.3 \
scikit-learn==1.7.0 \
scipy==1.15.3 \
six==1.17.0 \
sympy==1.14.0 \
threadpoolctl==3.6.0 \
tokenizers==0.21.1 \
torch==2.3.1 \
tqdm==4.67.1 \
transformers==4.48.2 \
triton==2.3.1 \
typing_extensions==4.14.0 \
tzdata==2025.2 \
urllib3==2.5.0 \
xxhash==3.5.0 \
yarl==1.20.1


RUN pip install deepod==0.4.1 --no-deps

# Verify installation
RUN python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Set default command
CMD ["python3"]

# Base Image (PyTorch2.4.1, CUDA 11.8)
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

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

# Install Python Dependency
RUN pip install \
tqdm==4.64.1
pandas==2.2.3
ydata-profiling==4.16.1
ipywidgets==8.1.7
pyod==2.0.5
python-dotenv==1.1.0

# Verify installation
RUN python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Set default command
CMD ["python3"]

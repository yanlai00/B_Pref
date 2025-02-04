# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]

##########################################################
### System dependencies
##########################################################

# Now let's download python 3 and all the dependencies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    libav-tools \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*


# Not sure why this is needed
ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && cp /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 -r \
    && rm mujoco.zip
COPY ./files/mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

# Not sure what this is fixing
COPY ./files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Not sure why this is needed
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

##########################################################
### Example Python Installation
##########################################################
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN conda create --name bpref python=3.6 pip
RUN echo "source activate bpref" >> ~/.bashrc
ENV OLDPATH $PATH
ENV PATH /opt/conda/envs/spt/bin:$PATH

# Install packages here
RUN conda install pytorch cudatoolkit=10.1 absl-py pyparsing
RUN conda install patchelf
RUN pip install mujoco_py
COPY ./docker_requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld

RUN pip install stable-baselines3[docs,tests,extra]
RUN pip install cloudpickle==1.3.0

# RUN pip install hydra-core --upgrade
# RUN pip install -e .[docs,tests,extra]
# RUN pip install -e custom_dmcontrol
# RUN pip install -e custom_dmc2gym

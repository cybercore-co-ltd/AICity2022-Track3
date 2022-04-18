ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install openmim and mmaction2 
RUN conda clean --all
RUN pip install openmim 
RUN mim install mmaction2==0.21.0

# Install AICity2022-Track3
RUN git clone https://github.com/cybercore-co-ltd/AICity2022-Track3.git /cctrack3
WORKDIR /cctrack3
RUN mkdir -p /cctrack3/data/raw_video
ENV FORCE_CUDA="1"
RUN python setup.py develop


# # Change conda owner
# USER root
# RUN groupadd conda 
# RUN chgrp -R conda /opt/conda
# RUN chmod a+rw /opt/conda

# RUN pip install cython --no-cache-dir
# RUN pip install --no-cache-dir -e .

# # Install mmcv-full
# RUN pip install mmcv-full==latest -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# # Install MMAction2
# RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
# WORKDIR /mmaction2
# RUN mkdir -p /mmaction2/data
# ENV FORCE_CUDA="1"
# RUN pip install cython --no-cache-dir
# RUN pip install --no-cache-dir -e .
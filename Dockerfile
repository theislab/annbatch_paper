FROM nvcr.io/nvidia/pytorch:25.12-py3

COPY . /opt/annbatch_paper
WORKDIR /opt/annbatch_paper
RUN pip install -e .

RUN pip install \
    jupyterlab \
    ipywidgets \
    ipykernel \
    lamindb \
    scDataset \
    annbatch["zarrs", "torch"] \
    anndata \
    scanpy \
    click \

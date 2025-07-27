FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN pip install --upgrade --no-cache \
    leidenalg==0.10.2 matplotlib==3.10.3 networkx==3.5 numpy==2.2.6 pandas==2.3.1 python_igraph==0.11.9 python_louvain==0.16 scanpy==1.11.3 scikit_learn==1.7.1 scipy==1.15.3 umap_learn==0.5.9.post2

# Install CellDiffusion
RUN pip install --upgrade --no-cache celldiffusion

# Install Jupyter
RUN pip install --upgrade --no-cache jupyterlab notebook

RUN mkdir /app
WORKDIR /app
COPY . .

CMD ["jupyter lab --ip='0.0.0.0' --allow-root"]

EXPOSE 8888

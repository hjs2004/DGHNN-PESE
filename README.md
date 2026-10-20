# DGHNN
This project developes DGHNN: Deep graph and hypergraph neural network for pan-cancer related gene prediction

![Figure 1](https://github.com/user-attachments/assets/9ae62daa-f3ce-48ed-ae02-b34c8b65c113)


# Data availability
The datasets used can be downloaded from https://owww.molgen.mpg.de/~sasse/EMOGI/
The file Homo_sapiens.gene_info can be downloaded from https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/Homo_sapiens.gene_info.gz

# Requirements
* Python 3
* PyTorch
* torch-geometric
* networkx
* captum
* pandas
* numpy
* sklearn

# Run
After installing the required environment and downloading the dataset, replace the corresponding file path in train.py, and run
```python
python train.py
```

## Node/edge positional and structural encodings

Training now enables GraphGPS-inspired node LapPE/RWSE and relative edge
encodings by default. See [GRAPH_ENCODINGS.md](GRAPH_ENCODINGS.md) for definitions,
dimensions, caching, the `DGHNN_ENCODINGS=0` baseline switch, and the bounded
`smoke_graph_encodings.py` check.

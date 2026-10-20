# Runtime profiles and deployment

The same source tree supports the local Windows workstation and the Linux GPU
server. PE/SE is enabled by default. Hardware settings are selected at runtime
and can be overridden explicitly; do not substitute workstation memory settings
when deploying to the server.

| Setting | Local/small GPU | Validated server profile |
|---|---|---|
| `DGHNN_PROFILE` | `local` | `server` |
| Edge chunk default | 32768 | 1048576 |
| Classifier batch size | 1000 | 1000 |
| AMP on CUDA | FP16 | FP16 |
| Block gradient checkpointing on CUDA | Enabled | Enabled |
| Edge-message recomputation for PE/SE | Enabled during training | Enabled during training |

`auto` selects the server profile on Linux with a CUDA GPU of at least 20 GiB.
The server launch script also explicitly selects the server profile and retains
its OMP/MKL thread settings. Hardware-specific launch scripts should be compared
and preserved during upgrades.

Before deploying, back up the existing source and compare the runtime section
of `train.py`. Stage new files separately, verify archive hashes, then replace
only the intended modules. Keep original datasets, environments and results.

Validate the deployed source before starting a long queue:

```bash
python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python smoke_graph_encodings.py --nodes 1024 --epochs 2 --device cuda
```

GPU 5 is only an example. Select a currently idle GPU instead of assuming its
availability. The smoke test is bounded and does not establish full-graph memory
capacity or predictive quality. See [GRAPH_ENCODINGS.md](GRAPH_ENCODINGS.md) for
encoding definitions and configuration.

The repository contains code, tests and setup documentation. Datasets, model
weights, logs, credentials, paper PDFs and local backups are excluded.

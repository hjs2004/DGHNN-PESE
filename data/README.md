# DGHNN datasets

Downloaded on 2026-09-17 from the data sources linked by the DGHNN paper and project README.

Paper: https://doi.org/10.1093/bioinformatics/btaf379

Preprocessed datasets: https://owww.molgen.mpg.de/~sasse/EMOGI/

| Network | Local training data |
| --- | --- |
| CPDB | `CPDB_multiomics.h5` |
| IRefIndex | `IREF_multiomics.h5` |
| IRefIndex 2015 | `IREF_2015_multiomics.h5` |
| Multinet | `MULTINET_multiomics.h5` |
| PCNet | `PCNET_multiomics.h5` |
| STRINGdb | `STRINGdb_multiomics.h5` |

The original compressed archives are in `archives/`. Only the HDF5 training datasets were extracted from the EMOGI model archives; archived model code was not executed. These datasets contain networks, multiomics features, gene identifiers, labels, and dataset splits.

`Homo_sapiens.gene_info` is decompressed from https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/Homo_sapiens.gene_info.gz . This is the current NCBI annotation downloaded on the date above, not an archived annotation version from the original experiments.

The repository already supplies `../KEGG_pathway_gene.txt`.

## Using these files

The original `train.py` has placeholder `/xxxxxxxx/` paths. Set its six dataset paths to the files above, its mapping path to `data/Homo_sapiens.gene_info`, and its pathway path to the repository's `KEGG_pathway_gene.txt`. The placeholder `PENCT_multiomics.h5` is a typo; the downloaded filename is `PCNET_multiomics.h5`.

The training code has not been modified or executed as part of the dataset download.

## Validation

See `download_manifest.json` for source URLs, archive SHA-256 digests, file sizes, and HDF5 dataset shapes. The validation checks gzip CRCs, expected EMOGI archive lengths, extracted sizes, required HDF5 fields, matching node counts, and sample reads. Local SHA-256 digests provide a record for future comparisons; the server does not supply independent reference digests here.

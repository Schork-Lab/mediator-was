# Mediator-was

A python project for conducting mediator-wide association studies. Currently, limited to TWAS, but techniques are generalizable to other endophenotypes. Accompanying preprint: Bhutani K, Sarkar A, Park Y, Kellis M, Schork NJ. “Modeling prediction error improves power of transcriptome-wide association studies”. bioRxiv. 2017. http://biorxiv.org/content/early/2017/02/14/108316


## Getting Started

### Prerequisites

* Numpy
* Scipy
* Pandas
* Matplotlib
* PyMC3
* FastLMM

### Installing

`pip install git+git://github.com:Schork-Lab/mediator-was.git#egg=mediator-was`

### Downloading Gene Models

We are currently working on providing access to our trained transcriptional regulation gene models.

## Usage
`mediator-was-twas --gene path_to_gene_directory --study path_to_study_prefix --out out_file`

| Argument    |   Description   |
| --------    | ------------------ |
| `--gene`    | Path to gene directory |
| `--study`   | Path to study vcf prefix  |
| `--out`     | Path to out files |
| `--rlog`    | Use RLog transformed values, default |
| `--gtex`    | Use GTEx-Norm values |
| `--min_inclusion` | Minimum inclusion probability for BAY-TS, default: 0.5 |
| `--max_missing` | Maximum missing genotypes for exclusion for study genotypes, default: 0.1 |

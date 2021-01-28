# phrase-embedding-meaningfulness

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
![python (scoped)](https://img.shields.io/badge/python-3.8.5-brightgreen.svg)

## Description

`phrase-embedding-meaningfulness` explores the application of a deep neural network to classify phrase meaningfulness from phrase embeddings.

## Usage

### Environment

Navigate to the `phrase-embedding-meaningfulness` directory and setup a new `conda` environment using the following commands.

```
conda create -n pem python=3.8.5 -y
conda activate pem
conda install ipykernel -y
ipython kernel install --user --name=pem
```

### Dependencies

Install the dependencies using the following command.

`pip install -r requirements.txt`

### Execution

To train and test the classifier, run the cells of the `classifier.ipynb` Jupyter notebook, using `jupyter lab`, ensuring the `pem` kernel is selected.

Run pipeline with `sudo`, as follows: `sudo jupyter lab --allow-root`. This is required to use `stanford-corenlp`, which is used in the pipeline and to save `Word2Vec` models.

## Classification

### Ablation Study

| Window Size | Embedding Vector Size | Number of Epochs | Batch Normalization | Number of Layers | Minimum Frequency | Test Accuracy |
| ----------- | --------------------- | ---------------- | ------------------- | ---------------- | ----------------- | ------------- |
| 5           | 100                   | 50               | Yes                 | 2                | None              | 0.61          |
| 5           | 100                   | 250              | Yes                 | 2                | None              | 0.64          |
| 5           | 100                   | 100              | Yes                 | 2                | None              | 0.58          |
| 5           | 100                   | 150              | Yes                 | 2                | None              | 0.65          |
| 5           | 200                   | 150              | Yes                 | 2                | None              | 0.66          |
| 5           | 300                   | 150              | Yes                 | 2                | None              | 0.62          |
| 5           | 200                   | 150              | No                  | 2                | None              | 0.69          |
| 5           | 200                   | 250              | No                  | 2                | None              | 0.66          |
| 5           | 200                   | 200              | No                  | 2                | None              | 0.65          |
| 5           | 200                   | 100              | No                  | 2                | None              | 0.71          |
| 5           | 200                   | 80               | No                  | 2                | None              | 0.74          |
| 5           | 200                   | 60               | No                  | 2                | None              | 0.72          |
| 5           | 200                   | 70               | No                  | 2                | None              | 0.72          |
| 5           | 200                   | 80               | No                  | 3                | None              | 0.75          |
| 5           | 200                   | 200              | No                  | 5                | None              | 0.76          |
| 5           | 200                   | 400              | No                  | 5                | None              | 0.73          |
| 5           | 200                   | 80               | No                  | 3                | 5                 | 0.79          |
| 5           | 200                   | 200              | No                  | 5                | 5                 | 0.80          |
| 10          | 200                   | 200              | No                  | 5                | 5                 | 0.82          |
| 10          | 200                   | 200              | No                  | 5                | 2                 | 0.76          |
| 5           | 200                   | 200              | No                  | 5                | 5                 | 0.83          |

## Results

A random sampling of noun phrases deemed meaningful by the network are as follows:

```
document classification
clustering tasks
a new perspective
the vanishing
respect
arrays
resolutions
noisier
both modalities
game
```

A random sampling of noun phrases deemed not meaningful by the network are as follows:

```
aims
a challenging problem
signals
samples
performance
mislabeled
accurate predictions
advantages
areas
presence
```

## References

<a href="https://dl.acm.org/doi/10.1145/3340531.3412029">[1]</a>
Li Wang, W. Zhu, Sihang Jiang, Sheng Zhang, Keqiang Wang, Yuan Ni, Guotong Xie, and Y. Xiao (2020).
Mining Infrequent High-Quality Phrases from Domain-Specific Corpora
Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20), 1535-1544.

<a href="https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89">[2]</a>
Akshaj Verma (2020).
PyTorch \[Tabular\] - Binary Classification
Towards Data Science

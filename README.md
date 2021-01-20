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

Run pipeline with `sudo`, as follows: `sudo jupyter lab --allow-root`. This is required to use `stanford-corenlp`, which is used in the pipeline.

## References

<a href="https://dl.acm.org/doi/10.1145/3340531.3412029">[1]</a>
Li Wang, W. Zhu, Sihang Jiang, Sheng Zhang, Keqiang Wang, Yuan Ni, Guotong Xie, and Y. Xiao (2020).
Mining Infrequent High-Quality Phrases from Domain-Specific Corpora
Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20), 1535-1544.

<a href="https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89">[2]</a>
Akshaj Verma (2020).
PyTorch \[Tabular\] - Binary Classification
Towards Data Science
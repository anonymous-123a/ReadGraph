# ReadGraph

# The code for ReadGraph

This repo contains the source code for the ReadGraph model.

## programming language

Python and Pytorch with 2.0.0 or later

## Required libraries

```
python==3.8.10
pytorch==2.0.0
torch-geometric==2.5.3
dgl==2.1.0
tensorboard==2.12.0
transformers==4.42.4
networkx==3.0
einops==0.8.0
```

## Hardware info

NVIDIA A40 48G GPU

## Datasets info

### Digg

This dataset can be obtained from the paper [1]. It includes users and the stories they create, with nodes representing both users and stories. Edges between users indicate following relationships, while edges between users and stories represent voting interactions. We select the meta-paths "USU", and "UU", where 'U' denotes a user and 'S' denotes a story. The relationships include 'following' and 'voting,' along with their respective inverse relationships.

## Yelp 

This dataset can be obtained from [2]. It captures user reviews for various hotels. Nodes represent users and hotels, and directed edges reflect user reviews toward hotels. Each edge is labeled as either normal or abnormal. The meta-path selected is "UHU", with 'U' indicating a user and 'H' indicating a hotel. The relationships include 'review' and its inverse. This dataset contains 10.27% of edges that are labeled as anomalous.

## Amazon

The dataset can be obtained from [3]. It contains user reviews of items on the Amazon platform. Nodes represent users and items, while directed edges denote user reviews of items. We use the meta-path "UIU", where 'U' denotes a user and 'I' denotes an item. The relationships include 'review' and its inverse.

## Data preprocessing

Since the Digg and Amazon datasets do not include labels for anomalous edges, we follow the way outlined in [1] by randomly injecting anomalous labels to 10% of the edges in these two datasets for the training set. Given that real-world anomalies are relatively rare, the remaining edges in these datasets are considered normal. The Yelp dataset, however, already contains labeled anomalous edges, so no additional injection was performed. For evaluation purposes, we maintain the original anomalies in the training set while additionally adjusting the anomaly percentage in the testing set to 5%, as specified in [1]. To achieve this in the Yelp dataset, we randomly remove some of the originally labeled anomalous edges to adjust the anomaly percentage to 5%.

## How to run the code for ReadGraph

```
python main.py --dataset=yelp
```

Main arguments:

```
--dataset [digg,yelp,amazon]: the name of dataset to run
--memory_num: the number of prototypes
--layers_num_M: the number of prototypes layers
--neighbour_num: the number of nodes in the sampled heterogeneous subgraphs
--snap_size: snapshot size
--window_size: the number of snapshots
```

For more argument options, please refer to `utils.py`

## References

[1] Li Y, Zhu J, Zhang C, et al. THGNN: An Embedding-based Model for Anomaly Detection in Dynamic Heterogeneous Social Networks[C]//Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 2023: 1368-1378.

[2] https://www.kaggle.com/datasets/abidmeeraj/yelp-labelled-dataset/code

[3] https://snap.stanford.edu/data/web-Amazon-links.html

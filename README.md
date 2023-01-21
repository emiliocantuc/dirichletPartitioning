# Dirichlet Partitioning
A simple Python's Numpy implementation of Dirichlet Partitioning.

In trying to replicate Section 3 of the *Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification* paper I implemented
a function that partitions an array of class labels using the [Dirichlet
distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) given a concentration
parameter (as I couldn't find the author's code). Thought it could be useful to others.

## Usage
```
import numpy as np
from dirichlet import dirichlet_partition

# An example of a class-balanced y
y=np.random.choice([0,1],size=10000,p=[0.5,0.5])

# Partitioned among 5 clients/segments with a high alpha
partition_indices=dirichlet_partition(y,c_clients=5,alpha=100,debug=True)

# partition_indices is now a dict. segment/client id -> y indices
# To get segment/client id -> values we can do
partition_values={i:y[indices] for i,indices in partition_indices.items()}

```

## Examples

```
>>> # A class balanced y
>>> y=np.random.choice([0,1],size=10000,p=[0.5,0.5])

>>> # Partitioned among 5 clients/segments with a high alpha
>>> partition=dirichlet_partition(y,c_clients=5,alpha=100,debug=True)

>>> summarize_partition(y,partition)
Client: 0 Counts: [ 912 1052] (total: 1964) %: [0.46 0.54]
Client: 1 Counts: [1043  926] (total: 1969) %: [0.53 0.47]
Client: 2 Counts: [ 930 1033] (total: 1963) %: [0.47 0.53]
Client: 3 Counts: [ 907 1055] (total: 1962) %: [0.46 0.54]
Client: 4 Counts: [1120  846] (total: 1966) %: [0.57 0.43]
Total samples: 9824


>>> # A class unbalanced y
>>> y=np.random.choice([0,1],size=10000,p=[0.9,0.1])

>>> # Partitioned among 5 clients/segments with a high alpha
>>> partition=dirichlet_partition(y,c_clients=5,alpha=100,debug=True)

>>> # Note how less total samples are distributed
>>> summarize_partition(y,partition)
Client: 0 Counts: [231 170] (total: 401) %: [0.58 0.42]
Client: 1 Counts: [176 223] (total: 399) %: [0.44 0.56]
Client: 2 Counts: [178 214] (total: 392) %: [0.45 0.55]
Client: 3 Counts: [199 200] (total: 399) %: [0.50 0.50]
Client: 4 Counts: [211 188] (total: 399) %: [0.53 0.47]
Total samples: 1990


>>> # A class balanced y
>>> y=np.random.choice([0,1],size=10000,p=[0.5,0.5])

>>> # Partitioned among 5 clients/segments with a low alpha
>>> partition=dirichlet_partition(y,c_clients=5,alpha=0.1,debug=True)

>>> # Note how low alphas produce unbalanced partitions
>>> summarize_partition(y,partition)
Client: 0 Counts: [   8 1270] (total: 1278) %: [0.01 0.99]
Client: 1 Counts: [  12 1256] (total: 1268) %: [0.01 0.99]
Client: 2 Counts: [1213] (total: 1213) %: [1.00]
Client: 3 Counts: [4966    8] (total: 4974) %: [1.00 0.00]
Client: 4 Counts: [1239] (total: 1239) %: [1.00]
Total samples: 9972


>>> # A class unbalanced y
>>> y=np.random.choice([0,1],size=10000,p=[0.1,0.9])

>>> # Partitioned among 5 clients/segments with a low alpha
>>> partition=dirichlet_partition(y,c_clients=5,alpha=0.1,debug=True)

>>> # Note how low alphas produce unbalanced partitions
>>> # and that y's baseline unbalance reduces total samples
>>> summarize_partition(y,partition)
Client: 0 Counts: [321] (total: 321) %: [1.00]
Client: 1 Counts: [516] (total: 516) %: [1.00]
Client: 2 Counts: [472  20] (total: 492) %: [0.96 0.04]
Client: 3 Counts: [321] (total: 321) %: [1.00]
Client: 4 Counts: [326] (total: 326) %: [1.00]
Total samples: 1976
```
Code used to produce examples is in the  `examples.ipynb` notebook.

## References
- Hsu, Tzu-Ming Harry, et al. Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification. arXiv, September 13, 2019. arXiv.org, [http://arxiv.org/abs/1909.06335](http://arxiv.org/abs/1909.06335).

# Deep Link Prediction for Graphical Database

## 1. Motivation and Goal
In recent years, deep learning has achieved great success in the fields of vision, speech, and natural language understanding. The ability of deep learning to extract underlying patterns from complex, large-scale and high-dimensional data is well recognized. Many real-world applications are built on a graphical database, and thus utilize graphs for storing data. The graph here means a directed, attributed multi-graph with data stored as nodes and relationships (links) between nodes stored as edges. Graphical databases are ubiquitous and often consist of hundreds of millions of nodes and relationships. There is rich information embedded in the complex topology of these graphs that can be leveraged when doing inference on the data stored in them. As a result, utilizing deep learning to extract this information has gained a lot of traction in the research community. 

In order to address the aforementioned problem, we followed two distinct approaches. In the first approach, we propose a Graph-Based Classification Model using a Convolutional Neural Network (CNN) that uses nodal features as well as the structure of a node's local sub-graph to predict links between graph nodes by using an adjacency matrix formulation. In the second approach, we propose a Graph Convolutional Neural Network (GCNN) which has a structure similar to an autoencoder. This formulation enables us to use our prior information about the structure of the graph more efficiently by putting a relational inductive bias [1] into our model.


## 2. Related Work
Neural networks that operate on graphs, and structure their computations accordingly, have been
developed and explored extensively for more than a decade under the umbrella of “graph neural
networks” [5], but have grown rapidly in scope and popularity in recent years.

Models in the graph neural network family, e.g. [9], have been explored in a diverse range of problem
domains, across supervised, semi-supervised, unsupervised, and reinforcement learning settings.
These models have been effective at tasks thought to have rich relational structure, such as visual
scene understanding tasks [10] and few-shot learning [3]. They have also been extensively used to
reason about knowledge graphs [2, 6]. For more applications of graph neural network models, see [1]
and references therein.

Recently, [4] introduced the message-passing neural network (MPNN), which unified various previous
graph neural network and graph convolutional network approaches by sequentially updating edge
attributes, node attributes and global attributes of the graph and transmitting the information after
each update. In a similar vein, [11] introduced the non-local neural network (NLNN), which unified
various “self-attention”-style methods by analogy to methods from computer vision and graphical
models for capturing long range dependencies in signals.

Graph neural network models usually consist of Graph Network (GN) blocks and can be divided into
three main categories depending on the task that needs to be served. Node-focused and graph-focused
GNs use the nodes attributes and the global attributes as outputs respectively. On the other hand, in
the spirit of [8, 7], our main scope in this project is the design of an edge-focused neural network in
order to predict the existence of an edge between two nodes as well as its corresponding label.


## References
1. P. W. Battaglia et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:
1806.01261, 2018.
2. M. Gori, G. Monfardini, and F. Scarcelli. A new model for learning in graph domains. In International
Joint Conference on Neural Networks, 2005.


__________________________________________________________________________________________________________________________________
How To Run:

___________________________________________________________________________________________________________________________________
Authors:

Aristotelis-Angelos Papadopoulos: aristotp@usc.edu

Collin Purcell					        : collinpu@usc.edu

Devershi Purohit				        : dupurohi@usc.edu

Ishank Mishra					          : imishra@usc.edu 

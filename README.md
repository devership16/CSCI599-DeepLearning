# Deep Link Prediction for Graphical Database

## 1. Motivation and Goal
In recent years, deep learning has achieved great success in the fields of vision, speech, and natural language understanding. The ability of deep learning to extract underlying patterns from complex, large-scale and high-dimensional data is well recognized. Many real-world applications are built on a graphical database, and thus utilize graphs for storing data. The graph here means a directed, attributed multi-graph with data stored as nodes and relationships (links) between nodes stored as edges. Graphical databases are ubiquitous and often consist of hundreds of millions of nodes and relationships. There is rich information embedded in the complex topology of these graphs that can be leveraged when doing inference on the data stored in them. As a result, utilizing deep learning to extract this information has gained a lot of traction in the research community. 

In order to address the aforementioned problem, we followed two distinct approaches. In the first approach, we propose a Graph-Based Classification Model using a Convolutional Neural Network (CNN) that uses nodal features as well as the structure of a node's local sub-graph to predict links between graph nodes by using an adjacency matrix formulation. In the second approach, we propose a Graph Convolutional Neural Network (GCNN) which has a structure similar to an autoencoder. This formulation enables us to use our prior information about the structure of the graph more efficiently by putting a relational inductive bias [1] into our model.


## 2. Related Work
Neural networks that operate on graphs, and structure their computations accordingly, have been
developed and explored extensively for more than a decade under the umbrella of “graph neural
networks” [2], but have grown rapidly in scope and popularity in recent years.

Models in the graph neural network family, e.g. [3], have been explored in a diverse range of problem
domains, across supervised, semi-supervised, unsupervised, and reinforcement learning settings.
These models have been effective at tasks thought to have rich relational structure, such as visual
scene understanding tasks [4] and few-shot learning [5]. They have also been extensively used to
reason about knowledge graphs [6,7]. For more applications of graph neural network models, see [1]
and references therein.

Recently, [8] introduced the message-passing neural network (MPNN), which unified various previous
graph neural network and graph convolutional network approaches by sequentially updating edge
attributes, node attributes and global attributes of the graph and transmitting the information after
each update. In a similar vein, [9] introduced the non-local neural network (NLNN), which unified
various “self-attention”-style methods by analogy to methods from computer vision and graphical
models for capturing long range dependencies in signals.

Graph neural network models usually consist of Graph Network (GN) blocks and can be divided into
three main categories depending on the task that needs to be served. Node-focused and graph-focused
GNs use the nodes attributes and the global attributes as outputs respectively. On the other hand, in
the spirit of [10,11], our main scope in this project is the design of an edge-focused neural network in
order to predict the existence of an edge between two nodes as well as its corresponding label.


## 3. Dataset 
We have used a combined dataset consisting of Bloomberg Bankruptcy Data for Companies (Year: 2009-2019) and a graph dataset containing approximately 6 million nodes (Person, Organizations, etc) with 10 million relationships.

<p align="center">
    <img src="figure/GraphExample.png" height="350"/>
</p>

Our graph dataset contains 8 types of nodes and 14 types of edges. Our goal was to initially create a super-node called Bankruptcy node and connect all US companies which filed bankruptcy according to the Bloomberg financial data. Then, by using the Image Segmentation model approach and the GCNN approach, we tried to predict the existence of an edge between a Company node and the Bankruptcy super-node.
It is worth mentioning the high class-imbalance problem we faced since the graph dataset contained $$>600,000$$ companies out of which only $$~3,500$$ companies had filed bankruptcy according to the collected Bloomberg financial data. 


## References
1. P. W. Battaglia et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:
1806.01261, 2018.
2. M. Gori, G. Monfardini, and F. Scarcelli. A new model for learning in graph domains. In International
Joint Conference on Neural Networks, 2005.
3. Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel. Gated graph sequence neural networks. In International
Conference on Learning Representations (ICLR), 2016.
4. A. Santoro, D. Raposo, D. G. Barrett, M. Malinowski, R. Pascanu, P. Battaglia, and T. Lillicrap. A simple
neural network module for relational reasoning. In Advances in Neural Information Processing Systems, 2017.
5. V. Garcia and J. Bruna. Few-shot learning with graph neural networks. In International Conference on
Learning Representations (ICLR), 2018.
6. A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko. Translating embeddings for
modeling multi-relational data. In Advances in Neural Information Processing Systems, pages 2787–2795, 2013.
7. T. Hamaguchi, H. Oiwa, M. Shimbo, and Y. Matsumoto. Knowledge transfer for out-of-knowledge-base
entities: A graph neural network approach. In International Joint Conference on Artificial Intelligence
(IJCAI), 2017.
8. J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for quantum
chemistry. arXiv preprint arXiv: 1704.01212, 2017.
9. X. Wang, R. Girshick, A. Gupta, and K. He. Non-local neural networks. In Proceedings of the Conference
on Computer Vision and Pattern Recognition (CVPR), 2017.
10. J. Hamrick, K. Allen, V. Bapst, T. Zhu, K. McKee, J. Tenenbaum, and P. Battaglia. Relational inductive
bias for physical construction in humans and machines. In Proceedings of the 40th Annual Conference of
the Cognitive Science Society, 2018.
11. T. Kipf, E. Fetaya, K.-C. Wang, M. Welling, and R. Zemel. Neural relational inference for interacting
systems. In Proceedings of the International Conference on Machine Learning (ICML), 2018.


__________________________________________________________________________________________________________________________________
How To Run:

___________________________________________________________________________________________________________________________________
Authors:

Aristotelis-Angelos Papadopoulos: aristotp@usc.edu

Collin Purcell					        : collinpu@usc.edu

Devershi Purohit				        : dupurohi@usc.edu

Ishank Mishra					          : imishra@usc.edu 

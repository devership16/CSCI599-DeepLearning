
## Motivation

<p align="center">
    <img src="figure/graph1.gif" height="250"/>
</p>

In recent years, deep learning has achieved great success in the fields of vision, speech, and natural language understanding. The ability of deep learning to extract underlying patterns from complex, large-scale and high-dimensional data is well recognized. Many real-world applications are built on a graphical database, and thus utilize graphs for storing data. The graph here means a directed, attributed multi-graph with data stored as nodes and relationships (links) between nodes stored as edges. Graphical databases are ubiquitous and often consist of hundreds of millions of nodes and relationships. There is rich information embedded in the complex topology of these graphs that can be leveraged when doing inference on the data stored in them. Our goal is to utilize deep learning models to extract this information and predict bankruptcies. 

In order to address the aforementioned problem, we followed two distinct approaches. In the first approach, we propose a Graph-Based Classification Model using a Convolutional Neural Network (CNN) that uses nodal features as well as the structure of a node's local sub-graph to predict links between graph nodes by using an adjacency matrix formulation. In the second approach, we propose a Graph Convolutional Neural Network (GCNN) which has a structure similar to an autoencoder. This formulation enables us to use our prior information about the structure of the graph more efficiently by putting a relational inductive bias [1] into our model.


## Related Work
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
models for capturing long-range dependencies in signals.

Graph neural network models usually consist of Graph Network (GN) blocks and can be divided into
three main categories depending on the task that needs to be served. Node-focused and graph-focused
GNs use the nodes attributes and the global attributes as outputs respectively. On the other hand, in
the spirit of [10,11], our main scope in this project is the design of an edge-focused neural network in
order to predict the existence of an edge between two nodes as well as its corresponding label.


## Dataset 
Using CrunchBase’s enterprise API we scraped approximately 5 million nodes and 8 million edges that model the U.S. financial system with an emphasis on early stage companies. Node types included Organization, Person, IPO, Funding Round, Job, Website, News, Address, Picture, Acquisition, Category, Fund, and Investment. Edge labels include such relationships as Employs, Was Founded By, Acquired, Invested In, Employed, Funded, Board of Director, and several more for a total of 29 unique relationship labels. See the figure below.

<p align="center">
    <img src="figure/GraphExample.png" height="350"/>
    <p>Fig. 1. Example sub-graph containing company, people, funding rounds, and product areas. Notice the highly interconnected nature of the nodes and variety of node types. </p>
</p>

In Fig. 1, the red nodes are companies and the small multi-colored nodes connected to them represent their Jobs, Persons, Funding Rounds, etc. To store this dataset we instantiated a NEO4J database and to query it we used py2neo; a python wrapper that can call NEO4J’s JAVA based cypher query engine. 

We collected our list of bankrupt companies using Bloomberg’s terminal API. To link the bankruptcies we found with the companies in the database we used simple string augmentation techniques like removing all co, org, corp, ltd, ect; all trailing ‘s’ characters; all spaces; and lowering all words. We choose to not use any string similarity metrics like Jaro or Levenshtein because we determined that even with a match threshold of 0.98 there was too much risk of false positives and injecting noise into our model. Using our techniques we found ~3.5k of the ~650k companies in our database had a match in the list of bankrupt companies scraped from Bloomberg. 

To integrate these matches into the graph in a way that conformed with our edge prediction problem formulation and topology sensitive models, we decided to create a single bankruptcy node and add a ‘Went_Bankrupt’ edge between it and every matched company. This linked every bankrupt company via the bankruptcy node and created rich adjacency matrices with multiple examples of bankrupt companies in each when predicting if an edge existed between the bankruptcy node and some new company in question. 

For licensing reasons we had to augment the aforementioned graph to obscure any identifiable information since we borrowed the data from one of our group member’s work group. To do this we embedded all nodal features using word2vec, TFiDF, and SVD. We also replaced all company names with a unique id. For memory and sparsity considerations we also converted several node types into edge features or, like in the case of Picture and Website, we dropped several node types altogether. For example, previously a graph walk could have included the series Picture<-Organization->Job->Person. After these conversions that walk would be Organization->Person with the features of that job as edge features. These conversions decreased the total number of node types to 8, number of nodes to 1.56M, number of edges to 3.14M, and edge types to 14.   

## Approaches 

### I. Image Segmentation Model

<p align="center">
    <img src="figure/ImageSegModel.JPG" height="250"/>
    <p align="center">Fig. 2. Auto-encoder architecture for the Image Segmentation Model(SegNet)</p>
</p>


#### Creating Adjacency Matrices


The highly interconnected nature of the financial sector makes the topology of our graph very dense and several nodes that we use to segment the graph, like product verticals and locations, have degrees of tens to hundreds of thousands. Therefore, we relax the requirement to take all of a company’s edges in exchange for a "good guess" of the most relevant nodes and improved latency and memory savings. To do this we chose the approximate nearest neighbor (ANN) library ANNOY and trained a unique index for each high degree node. We created node features using a word2vec model pre-trained on the Wikipedia corpus and TFiDF to embed the words in the nodal descriptions and do a weighted sum of their vectors. We found that we needed both word2vec and TFiDF; word2vec to ensure the semantic similarity of the descriptions and TFiDF to force keyword similarities. We then appended normalized and scaled nodal features that describe the number of employees, funding, and valuation. 


To integrate ANNOY, which is C based with a python wrapper, natively into NEO4js cypher API, which is written in JAVA, we used a JAVA wrapper with limited functionality. We then wrote our own User-Defined Function (UDF) NEO4j module that calls the java wrapper for ANNOY which then returns the node ids for the N nearest neighbors to the query node for each high degree node that it shares an edge with. The final output of our query is the union of those N nodes per high degree node and their one-hop neighbors, up to 50 per high degree node, and the three-hop distance neighbors that are not connected to a high degree node from the query node. This gave us a balance between the nodes in the query node’s local network and a global context about the financial space the query node is in. We then took this subgraph, transformed it into a 2D adjacency matrix, padded or cropped until it was 240x240, and added the nodal features in the third dimension after passing them through an SVD model. More concretely, when two nodes i and j shared an edge we concatenated their nodal features and inserted that into the cell [i,j] in the adjacency matrix. Finally, we also included negative sampling by selecting two edges in the adjacency matrix and swapping their head and tail nodal features. The output matrix was the same 240x240 adjacency matrix with the one hot encoding for the edge labels in the third dimension. 


#### Model Architecture

We used a modified SegNet model written in Keras to map the input 3D adjacency matrices with nodal features to the output 3D adjacency matrices with class distributions. To conform SegNet to our data we changed how the data was accessed throughout the model and also prepended a convolutional layer with 32 1x1 filters. This layer allowed the model to explicitly aggregate the concatenated nodal features in a learned way before they are passed to the 3x3 kernels in the first layer of SegNet.

#### Training

We precomputed training samples due to neo4j and disk bottlenecks and used a Keras generator to asynchronously load them into memory during training in batch sizes of 10. For visualization and reporting we created custom precision and recall metrics and custom tensorboard logging functions. Initially, we also included an early stopping callback based on validation loss, but we found that even with large patience values this would consistently stop training the model too early so we removed it. Finally, to account for the massive class imbalance due to edge sparsity in the adjacency matrix as well as bankrupt versus non-bankrupt companies, we hand tailored class weights to bias the model’s loss function and take these imbalances into account. 




### II. Graph Convolutional Neural Network Model
Our approach to building a model using Graph Convolutional Neural Network (GCNN) to solve the multi-relational link prediction task in a multimodal finance graph, inspired by [12], had to take care of an important observation relating to the nature of the dataset. There is a huge variation in the number node pairs that the data set contains corresponding to each edge type. Therefore, it becomes important that we develop an end-to-end approach such that the model shares the parameters from different edge types. 

In our approach, we build a non-linear, multi-layer neural network model designed to operate on a graph. The model has two main components:

- **Encoder:** The objective is to produce an embedding for each node in the graph
- **Decoder:** A model that uses these embeddings to perform tensor factorization and predicts the edges

We built an end-to-end model where the node embeddings are optimized jointly along with tensor factorization
We describe both the encoder and decoder in detail

#### GCNN Encoder  
The input to the encoder is the nodal feature vectors $$h_i$$, and the graph $$G = (V, R)$$ with nodes $$v_i \in V$$ and labeled edges $$(v_i, r, v_j)$$ where $$r \in R$$ is an edge type.  The output is a d-dimensional embedding $$h_{i}^{k+1}$$ for each node. 

<p align="center">
    <img src="figure/GCN.JPG" height="450"/>
    <p align="center">Fig. 3. A two-layer multimodal network for GCNN Model with convolutions across K-hop distances</p>
</p>

For a given node, the model takes into account the feature vector of its first-order neighbors. Since each neighbor can be of a different node type and can have different edge label, we have a different neural network architecture for each node. Each node type can have different lengths of embeddings; therefore, it is important that each edge type has a different set of weights. Note, an edge type is different if the node types are reversed. The convolution operators we define in the encoder uses these weights depending on the neighbors and edge types. On the successive application of these convolution operators, we essentially convolve across a K-hop distance in the graph for each neighbor. In other words, each node’s embeddings would have been formed using the information passed from all it’s Kth-order neighbors while taking into account the different edge types [13]. This is depicted in Fig. 3, which shows convolutions around a node. A single convolution on the neural network takes the following form: 

##### $$h_{i}^{k+1} = \phi(\sum_r \sum_{j \epsilon N_r^i} c_r^{ij} W_r^k h_j^k + c_r^i h_i^k)$$  

Where $$h_i^k$$ the embedding of node $$v_i$$ in the kth layer with a dimensionality $$d^k$$, r is an edge type and $$W_k^r$$ is a weight/parameter matrix corresponding to it, $$\phi$$ represents a non-linear activation function, $$c_r^{ij}$$ are normalization constants. A more detailed view of one convolution for the encoder is depicted in Fig. 4. We build a two-layer model by stacking two layers of these. The input to the first layer is the node feature vectors or one-hot vectors if the features are not present.

<p align="center">
    <img src="figure/GCNEncoder.JPG"/>
    <p align="center">Fig. 4. Encoder Architecture for the GCNN Model</p>
</p>

#### GCNN Decoder  
The input to the decoder is a pair of node embeddings that we want to decode. We treat each edge label differently i.e. we have a function $$g$$ that scores how likely it is that two nodes $$v_i$$ and $$v_j$$ have an edge type $$r$$ between them. 

##### $$p_r^{ij} = \sigma (g(v_i, r, v_j)) $$  


The decoder is a rank-d DEDICOM tensor factorization of a 3-way tensor [14,15]. We take the embeddings of two nodes produced by the encoder, $$z_i$$ and $$z_j$$, and use them as inputs to the decoder, which in turn predicts if an edge type $$r$$ exists between the nodes. This is depicted in Fig. 5. The decoder model can be mathematically described by the following equation: 

##### $$g(v_i, r, v_j) = z_i^T D_r R D_r Z_j $$  


where $$R$$ is a trainable weight matrix that models the global variations between the two node types $$i$$ and $$j$$. This parameter is shared between all the edge types corresponding to the node types. The other parameter is $$D_r$$, a diagonal matrix, which is used to map local interactions for each edge type $$r$$. It models the importance of each dimension in the node embeddings towards the prediction of the the existence of an edge type $$r$$. 

<p align="center">
    <img src="figure/GCNDecoder.JPG"/>
    <p align="center">Fig. 5. Decoder Architecture for the GCNN Model</p>
</p>

#### GCNN Training  
The trainable parameters of the model are 
- weight matrices for each edge type $$W_r$$
- Weight matrix $$R$$ for mapping interaction between two node types
- A diagonal weight matrix $$D_r$$ corresponding to each edge type. 

We have used the cross-entropy loss to optimize our model. The loss function can be written as: 

##### $$J_r(i, j) = -\log p_r^{ij} - \mathbb{E}_{n\sim P_r (j)} \log(1 - p_r^{in})$$
##### $$J = \sum_{(v_i, r, v_j) \in R} J_r(i, j)$$  


We have used negative sampling to estimate the model. For each edge type r between nodes $$v_i$$ and $$v_j$$ (positive sample), we choose another node $$v_n$$ randomly and sample the edge type $$r$$ (negative sample) between them [16]. 

The loss gradients are then propagated through the decoder and encoder, thus, performing an end-to-end optimization and jointly optimizing all the parameters.  We train our model for 50 epochs using the ADAM optimizer with a learning rate of 0.001. To initialize the weights, we use a method proposed by [17]. We also normalize the node features. We use sparse matrix multiplications due to the enormous size of the matrices which are quite sparse. We also apply dropout to the hidden layers to prevent overfitting and thus allowing the model to generalize well. 

We create batches by randomly selecting an edge type and then randomly picking edges from it. If the samples are exhausted then the edge type is not sampled again in the same epoch. If all the edge types are exhausted then we count that as one epoch. This way the edge types are picked in the order of their contribution to the loss function. This approach helps in fitting the model in memory.



## Results
### I. Image Segmentation Model (SegNet):

We ran the Image segmentation model for the whole data-set and observed the following results:

<p align="center">
    <img src="figure/SegNet_train.jpeg" height="300"/>
    <p align="center">Fig. 6. Training Loss, Accuracy, Bankruptcy Recall, and Bankruptcy Precision for SegNet Model</p>
</p>

<p align="center">
    <img src="figure/SegNet2_val.jpeg" height="300"/>
    <p align="center">Fig. 7. Validation Loss, Accuracy, Bankruptcy Recall, and Bankruptcy Precision for SegNet Model</p>
</p>





   **Table 1: Results of SegNet Model for the Bankruptcy Edge**



|       			| Accuracy  	| Recall     	| Precision 	| **AUPRC**     |
|:-----------------	|:-----------	|:-----------	|:-----------	|:-----------   |
| SegNet           	| 0.9757        | 0.6299        | 0.3571       	| **0.43**      |



**NOTE:** AURPRC - Average Precision Score








<p align="center">
    <img src="figure/SegNet_PRC.jpeg" height="250"/>
    <p align="center">Fig 8. Precision-Recall Curve for the SegNet Model </p>
</p>
<p align="center">
    <img src="figure/SegNet_all.jpeg" height="450"/>
    <p align="center">Fig 9. Recall distrubution for all edge types </p>
</p>






In the above confusion matrix, we see that the model performs well for most classes. The classes it does not perform well on, like Had_Funding_To, by design all had low-class weights so their lower performance was to be expected. This was done to keep more emphasis on edge labels that we deemed more relevant to predicting bankruptcies given the data. We also see in the precision-recall curve that the model has a precision of roughly 0.9 when you consider recall scores of 0.0-0.3. This level of precision is a terrific sign and indicates that the model provides near-certain predictions on roughly 30% of the validation bankruptcy samples. Another great sign is that by smoothing the training curves for the validation loss, bankruptcy recall, and bankruptcy precision, we see monotonically improving graphs indicating that even better performance is achievable with more training. 

In general, however, the adjacency matrix sparsity was too much to compensate for with just class weights and this model struggled to get higher than a 0.4 AUPRC. In a typical adjacency matrix, there would be 240 nodes and roughly 20% of those nodes would be companies. Since only 0.5% of companies went bankrupt there will be roughly 1 bankrupt company per every 4 adjacency matrices. This means that out of the 240x240 cells in the adjacency matrix there is a 1 in ~200,000 chance that a cell will have the label ‘Went_Bankrupt’. 



### II. Graph Convolutional Neural Network (GCNN):

We ran our GCN model for graphs with a different number of total nodes(10K, 20K, 25K, and 30K nodes) and observed the following results:




   **Table 2: Results of GCNN Model for all edge types (bi-directional) in the Graph**   
   
   
   
   

   
| Edge Type          	| AUPRC_10K 	| AUPRC_20K 	| AUPRC_25K 	| **AUPRC_30K** |
|:--------------------	|:-----------	|:-----------	|:-----------	|:-----------	|
| Employs            	| 0.6076    	| 0.6782    	| 0.70031   	| 0.69323   	|
| Employed           	| 0.56125   	| 0.6083    	| 0.60821   	| 0.62838   	|
| Was_Founded_By     	| 0.84427   	| 0.8729    	| 0.861     	| 0.87204   	|
| Employs            	| 0.85175   	| 0.9095    	| 0.97228   	| 0.97663   	|
| Employed           	| 0.81013   	| 0.9601    	| 0.96152   	| 0.96193   	|
| Was_Founded_By     	| 0.85535   	| 0.9046    	| 0.89107   	| 0.91841   	|
| Acquired           	| 0.79239   	| 0.9705    	| 0.97351   	| 0.9717    	|
| Was_Invested_In_By 	| 0.80587   	| 0.9489    	| 0.93843   	| 0.90202   	|
| Acquired           	| 0.78444   	| 0.9886    	| 0.97656   	| 0.99003   	|
| Was_Invested_In_By 	| 0.74629   	| 0.9259    	| 0.92427   	| 0.93148   	|
| **Went_Bankrupt**     | **0**         | **0.4906**    | **0.62563**   | **0.73012**   |







**NOTE:** AURPRC - Average Precision Score









<p align="center">
    <img src="figure/GCN_AUPRC_Bkrpt.PNG" height="350"/>
    <p align="center">Fig. 10. AUPRC Score for the Went_BankruptEdge over varying size of graphs </p>
</p>
<p align="center">
    <img src="figure/GCN_AUPRC_all.PNG" height="450"/>
    <p align="center">Fig. 11. AUPRC Score for all Edge Types over varying size of graphs</p>
</p>

We observe that in general, as the number of total nodes increases in the graph, the average precision score (AUPRC) increases for all the edge types.  The AUPRC score for the graph with 10K nodes is 0 as the number of nodes with the Went_Bankrupt edges is very less. As the total number of nodes increases in the graph, the number of organizations with the Went_Bankrupt edge also increases and hence the AUPRC score for the Went_Bankrupt edge also increases. We see that the graph with just 30K nodes achieves a fairly good AUPRC score of 0.73.


## Limitations
The finalized graph generated after our preprocessing still had many limitations. Due to these limitations, we faced several issues while running over models which are discussed as follows. 

- Because of our graph's emphasis on early-stage companies, many companies had 90% or more NaN values since they are not yet beholden to stock owners or SEC financial reporting regulations to publicly report metrics about their financial health. We hypothesize that there is likely very little predictive signal in these nodes and that they could possibly be injecting noise or biases into our models decreasing their performance. 

- Another shortcoming in the graph was its agnosticism to global financial metrics such as inflation, unemployment rate, GDP, ect., which could drastically affect a company's viability. For example, our models would be unaware that two potentially identical companies could have very different viabilities if one was started during the 2008 Great Recession versus today. 

- Another disappointment was that the News data, which would have likely been one of the richest data resources in the graph for predicting bankruptcies, were only links to articles coming from dozens of different sources. To scrape all these sources, mine relevant and accurate data from their articles, and intelligently integrate that data into the graph’s topology would have taken a huge effort and at this point is left as future work. 

- Finally, it stands to reason that more companies out of the 650k in our graph went bankrupt than the ~3.5k matches we found in the list we scraped from Bloomberg; especially given we matched from two separate data sources and CrunchBase’s early stage focus. All of these non-matched bankruptcies that are in the graph inject tremendous noise into our models by forcing them to model a class distribution that is not the true distribution and thusly potentially conflicting examples. Unfortunately correcting this problem, if possible, would likely require purchasing another API subscription from CrunchBase and which is beyond our capabilities at this point.

## Conclusions
In this project, we used a financial graph database in order to predct companies in the U.S. which went bankrupt. To this end, we proposed two distict approaches. In the first approach, we used an Image Segmentation model (SegNet) while in the second approach, we used a Graph Convolutional Neural Network (GCNN). In general, we observed that the average precision score for edge types increased as the number of nodes in the graph increased. Also, the GCNN model(**AUPRC: 0.73**) for a partial sample (30K nodes) achieved a better average precision score compared to the SegNet model(**AUPRC: 0.43**) running on the complete dataset. We also observed that the GCNN model avoided the issue of sparsity in adjacency matrices faced by the SegNet model, by considering only connected neighbors in the graph. The GCNN model also leveraged the graphical structure of our data-set by incorporating nodal features for the nodes and hence improved the results.

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
12. M. Zitnik, M. Agrawal, and J. Leskovec. Modeling polypharmacy side effects with graph convolutional networks. Bioinformatics, 2018.
13. M. Schlichtkrull, et al. Modeling relational data with graph convolutional networks. arXiv: 1703.06103, 2017.
14. M. Nickel, et al. A three-way model for collective learning on multi-relational data. In Proceedings of the International Conference on Machine Learning (ICML), vol. 11, pp. 809–816, 2011. 
15. T. Trouillon, et al. Complex embeddings for simple link prediction. In Proceedings of the International Conference on Machine Learning (ICML), vol. 33, pp. 2071–2080, 2016.
16. T. Mikolov, et al. Distributed representations of words and phrases and their compositionality. In NIPS, pp. 3111–3119, 2013.
17. X. Glorot, and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, vol. 13, pp. 249–256, 2010.




## Authors:

Aristotelis-Angelos Papadopoulos: aristotp@usc.edu

Collin Purcell					        : collinpu@usc.edu

Devershi Purohit				        : dupurohi@usc.edu

Ishank Mishra					          : imishra@usc.edu 

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>



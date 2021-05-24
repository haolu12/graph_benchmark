#########################
Rapids Benchmark Result:
#########################

1. Data Set:
**************************
This Data set will be used to benchmark all other libs.
.. image:: https://github.com/haolu12/graph_benchmark/blob/main/input.png

2. Single GPU results:
	Below is the exact run-time of various cuGraph kernels. For the sake of performance measurement, I have converted the directed graph into undirected graph. The non-zero will be total number bi-directional edgs*2.
.. image:: https://github.com/haolu12/graph_benchmark/blob/main/runtime.png

To better understand the performance, we pick to look at Million edges per seconds measurement, since most of the algorithm here are implemented with O(m) complexity.
.. image:: https://github.com/haolu12/graph_benchmark/blob/main/performance.png

Before the RMM memory limited is reached Load+Cons can achieve up to 17M edges/Sec. In my opinion RMM is having trouble due to the renumbering function.
Pagerank operations with decent size of graph and large enough degree per node can achieve up to 250M edges/Sec. However, banded graph like road network is not performing due to its limited parallelism. Louvain method is a vertex centric clustering method that has significant amount of parallelism. Cugraph can cluster 9M edges/Sec to 30M E/Sec when file is smaller than 250M edge. For BFS, the result showed here are actually an overestimate about 2x, since bi-directional edge in BFS should count as just 1 edge. User should be able to use this chart and estimate the runtime of their graph.
I my opinion cuGraph on single node is suite for fast turnaround development on small graph, however it is not the optimal solution. 

2. Multi GPU results:
	Current cugraph version (0.18) has limited support on multi gpus kernel. The result does not seem to be good for any kind of further investigation. The file loading and graph construction phase stop scaling after 4 workers. Pagerank is either issue with single GPU or it never scale beyond 1 gpu. Betweenness centrality is scaling well up to 5 workers; the chart may seem super linear scaling, but it does not. Different from Pangerank and Load+Cons, batch node also participated in the computation of Betweeness centrality, which mean 2 workers = 3 GPUs.
.. image:: https://github.com/haolu12/graph_benchmark/blob/main/distributed.png

Bug Notes (RMM):
	Random warning on no Nvidia GPU.

Bug Notes (BFS):
	The following error causing multi-GPUs BFS to fail. I am suspecting the serialization during communication causing the error. In addition, dask.cudf.series does not support sampling causing difficulty to random select starting node. Error code: 
	Traceback (most recent call last):
  File "./verify_dask_cuda_cluster.py", line 67, in <module>
    df = dcg.bfs(dg,0)
  ...
  File "cugraph/dask/traversal/mg_bfs_wrapper.pyx", line 106, in cugraph.dask.traversal.mg_bfs_wrapper.mg_bfs
TypeError: an integer is required
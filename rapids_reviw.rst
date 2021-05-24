#############
Heading 1
#############

*************
Heading 2
*************

===========
Heading 3
===========

Heading 4
************

Heading 5
===========

Heading 6


Running Rapids on Summit.

RAPIDS cuGraph is a library of graph algorithms that seamlessly integrates into the RAPIDS data science ecosystem and allows the data scientist to easily call graph algorithms using data stored in a GPU DataFrame, NetworkX Graphs, or even CuPy or SciPy.

1. Cloning Environment:
	module load python/3.7.0-anaconda3-5.3.0
	module load ums
	module load ums-gen119
	module load nvidia-rapids/0.18

	- IF you would like to install mpi4py:
	conda create --clone /sw/summit/ums/gen119/nvrapids_0.18_gcc_7.4.0--name myEnv
	source activate myEnv
	CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py


2. Single cuGraph Run:
	Single GPU cugraph is simple to use. It works like networkX and other graph libraries. The main parameter needs to pay attention is the rmm feature, when graph size is over 200M on 16GB GPU, it will be required to turn on rmm feature. #Spell out rmm

Example test.py for BFS:
import cugraph
import cudf
import time
from collections import OrderedDict
import random
import networkx as nx
from scipy.io import mmread
import sys
import rmm

print("input file:", sys.argv[1])
datafile = sys.argv[1]
direction = sys.argv[2]
rmm_flag = sys.argv[3]

if(rmm_flag == "y"):
	print("Initial RMM")
	rmm.reinitialize(managed_memory=True)
	assert(rmm.is_initialized())

G = cugraph.Graph()

if(direction == "D"):
	gdf = cudf.read_csv(datafile, names=["src", "dst"], delimiter=' ', dtype=["int32", "int32"] )
	G = cugraph.DiGraph()
	G.from_cudf_edgelist(gdf, source='src', destination='dst')

else:
	gdf = cudf.read_csv(datafile, names=["src", "dst"], delimiter=' ', dtype=["int32", "int32"] )
	G = cugraph.Graph()
	G.from_cudf_edgelist(gdf, source='src', destination='dst')

# 64 Random BFS
startVertex=G.nodes().sample(64, keep_index=False)
t0 = time.time()
for i in range(64):
	df = cugraph.bfs(G,startVertex[i])
t1 = time.time()
print("Cudf BFS takes: " + str(t1-t0))


(This is a copy from github.com/benjha/nvrapids_olcf)
Example LSF file:
#BSUB -P <PROJECT>
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -q batch
#BSUB -J rapids_test
#BSUB -o rapids_test_%J.out
#BSUB -e rapids_test_%J.out
module load ums
module load ums-gen119
module load nvidia-rapids/0.18
#what happened if not disable
jsrun -n 1 -a 1 -c 1 -g 1 --smpiargs="-disable_gpu_hooks" python PATH_EXE/test.py

3. Multi GPU Run: 
	In order for cuGraph to run on multi-GPUs, dask scheduling is required.
Procedure goes as follow:
#Add a link to dask.
	a. Build the cuGraph executable (test.py). 
		- The executable will need to access to dask schedulers information.
		- It is simple to pass to the executable as input argv.
	b. Build the LSF file for Dask work/scheduler
		- The lsf file will use 1 jsrun for scheduler, 1 jsrun for workers.
		- The executable will be launched on batch node, and the workload will be 					passed onto workers by using dask call.
	c. bsub test.lsf

Example test.py for Pagerank
import sys
from dask.distributed import Client
import cugraph
import dask_cudf
import cudf
import time
import cugraph.dask as dcg
import cugraph.comms as Comms
import rmm

def disconnect(client, workers_list):
	client.retire_workers(workers_list, close_workers=True)
	client.shutdown()

if __name__ == '__main__':
	sched_file = str(sys.argv[1]) #scheduler file
	num_workers = int(sys.argv[2]) # number of workers to wait for
    
	# 1. Connects to the dask-cuda-cluster
	client = Client(scheduler_file=sched_file)
	print("client information ",client)

	# 2. Blocks until num_workers are ready
	print("Waiting for " + str(num_workers) + " workers...")
	client.wait_for_workers(n_workers=num_workers)
	workers_info=client.scheduler_info()['workers']
	connected_workers = len(workers_info)
	print(str(connected_workers) + " workers connected")

	# 3. Do computation
	rmm.reinitialize(managed_memory=True)
	assert(rmm.is_initialized())

	print("input file:" + str(sys.argv[3]))
	input_data_path=sys.argv[3]
	Comms.initialize(p2p=True)
	chunksize = dcg.get_chunksize(input_data_path)
	ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize, delimiter=' ', names=['src', 'dst'], dtype=['int32', 	'int32'])
	dg = cugraph.DiGraph()
	dg.from_dask_cudf_edgelist(ddf, 'src', 'dst')
	pr_df = dcg.pagerank(dg, tol=1e-4)
	
	# 4. Shutting down the dask-cuda-cluster
	print("Shutting down the cluster")
	workers_list = list(workers_info)
	disconnect (client, workers_list)
 

(This is a copy from github.com/benjha/nvrapids_olcf)
Example test.lsf for Launch Dask worker and Dask Scheduler:
#BSUB -P <PROJECT>
#BSUB -W 0:05
#BSUB -alloc_flags "gpumps smt4 NVME"
#BSUB -nnodes 2
#BSUB -J rapids_dask_test_tcp
#BSUB -o rapids_dask_test_tcp_%J.out
#BSUB -e rapids_dask_test_tcp_%J.out
module load ums
module lo fiad ums-gen119
module load nvidia-rapids/0.18
SCHEDULER_DIR=$PATH_TO_SCRATCH/dask_scheudler
WORKER_DIR=$PATH_TO_SCRATCH/dask_worker
if [ ! -d "$SCHEDULER_DIR" ]
then
    mkdir $SCHEDULER_DIR
fi
if [ ! -d "$WORKER_DIR" ]
then
    mkdir $WORKER_DIR
fi

SCHEDULER_FILE=$SCHEDULER_DIR/my-scheduler.json

jsrun -n 1 -a 1 -c 1 –smpiargs="-disable_gpu_hooks" dask-scheduler --interface ib0 \
		--scheduler-file $SCHEDULER_FILE --no-dashboard --no-show &
#Wait for the dask-scheduler to start
sleep 10
jsrun -r 6 -a 1 -c 2 -g 1 --smpiargs="-disable_gpu_hooks" dask-cuda-worker --nthreads 1 \ 
		--memory-limit 82GB --device-memory-limit 16GB --rmm-pool-size=15GB \
           		--death-timeout 60  --interface ib0 --scheduler-file $SCHEDULER_FILE \
		--local-directory $WORKER_DIR --no-dashboard &
#Wait for WORKERS
sleep 10
# This number will be nnode * -r
WORKERS=12	
python PATH_EXE/test.py $SCHEDULER_FILE $WORKERS
#clean DASK files
rm -fr $SCHEDULER_DIR
rm -rf $WORKER_DIR

Notes on Multi GPUs with cuGraph:
	Launching test.py with batch node.
	- I would recommend not to lunch test.py with batch node with cuGraph. Some of the 	graph feature require local processing with cudf and cugraph, example as G.batch().
	- Suggestions to lsf file with 2 nodes:
	#specify total number of workers instead of worker per node. Leave 1 gpu for the main function for test.py
	jsrun -n 11 -a 1 -c 2 -g 1 --smpiargs="-disable_gpu_hooks" dask-cuda-worker --nthreads 1 \ 
		--memory-limit 82GB --device-memory-limit 16GB --rmm-pool-size=15GB \
           		--death-timeout 60  --interface ib0 --scheduler-file $SCHEDULER_FILE \
		--local-directory $WORKER_DIR --no-dashboard &
	#Wait for WORKERS
	sleep 10
	# This number will be nnode * -r
	WORKERS=11	
	jsrun -n 1 -a 1 -c 1 -g 1 python PATH_EXE/test.py $SCHEDULER_FILE $WORKERS

For more information about running Rapids on Summit: https://github.com/benjha/nvrapids_olcf/blob/branch-0.19/docs/nvidia-rapids.rst
For more information about Rapids/cuGraph Lib:
https://docs.rapids.ai/api/cugraph/legacy/
https://github.com/rapidsai/cugraph/tree/branch-0.18


Benchmark Result:

1. Data Set:
	This Data set will be used to benchmark all other libs. T












		#Reference of data set suiteSparse, categories/enrich the description

2. Single GPU results:
	Below is the exact run-time of various cuGraph kernels. For the sake of performance measurement, I have converted the directed graph into undirected graph. The non-zero will be total number bi-directional edgs*2.

To better understand the performance, we pick to look at Million edges per seconds measurement, since most of the algorithm here are implemented with O(m) complexity.











Before the RMM memory limited is reached Load+Cons can achieve up to 17M edges/Sec. In my opinion RMM is having trouble due to the renumbering function.
Pagerank operations with decent size of graph and large enough degree per node can achieve up to 250M edges/Sec. However, banded graph like road network is not performing due to its limited parallelism. Louvain method is a vertex centric clustering method that has significant amount of parallelism. Cugraph can cluster 9M edges/Sec to 30M E/Sec when file is smaller than 250M edge. For BFS, the result showed here are actually an overestimate about 2x, since bi-directional edge in BFS should count as just 1 edge. User should be able to use this chart and estimate the runtime of their graph.
I my opinion cuGraph on single node is suite for fast turnaround development on small graph, however it is not the optimal solution. 

2. Multi GPU results:
	Current cugraph version (0.18) has limited support on multi gpus kernel. The result does not seem to be good for any kind of further investigation. The file loading and graph construction phase stop scaling after 4 workers. Pagerank is either issue with single GPU or it never scale beyond 1 gpu. Betweenness centrality is scaling well up to 5 workers; the chart may seem super linear scaling, but it does not. Different from Pangerank and Load+Cons, batch node also participated in the computation of Betweeness centrality, which mean 2 workers = 3 GPUs.







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



Source Code for benchmark and lsf file:
		In near future to host on github


#########################
Running Rapids on Summit
#########################

RAPIDS cuGraph is a python library of graph algorithms that seamlessly integrates into the RAPIDS data science ecosystem and allows the data scientist to easily call graph algorithms using data stored in a GPU DataFrame, NetworkX Graphs, or even CuPy or SciPy. For more introduction please visit `Rapids <https://github.com/rapidsai/cugraph>`_
In this documents we will introduce how to run cuGraph with Summit resources.


1. Cloning Environment:
**************************
Rapids environment had been installed on Summit.

::

  To use the library:

      module load python/3.7.0-anaconda3-5.3.0
      module load ums
      module load ums-gen119
      module load nvidia-rapids/0.18

::

 If you would like to install mpi4py:

     conda create --clone /sw/summit/ums/gen119/nvrapids_0.18_gcc_7.4.0--name myEnv
     source activate myEnv
     CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py




2. Single cuGraph Run:
**************************
Single GPU cuGraph is simple to use. It works just like networkX and other graph libraries. The main parameter needs to pay attention is the RMM feature (RAPIDS Memory Manager). When processing graph that contains more than 200 millions edges on 16GB GPU (Summit V100), RMM feature need to be on. 

Example python code:
::

  BFS.py

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


Example LSF file:
::

 batch.lsf 

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
     jsrun -n 1 -a 1 -c 1 -g 1 --smpiargs="-disable_gpu_hooks" python PATH_EXE/BFS.py

The ``--smpiargs`` flag must be disabled. Without disable, error of missing python binding will occur.

3. Multi GPU Run: 
**************************
In order for cuGraph to run on multi-GPUs, Dask scheduling is required. For more information on how dask work, please visit their page `Dask <https://github.com/dask/dask>`_

To procedure to run Dask and cuGraph goes as follow:

#. Build the cuGraph executable (test.py).
    - The executable must make a connection to Dask schedulers information.
    - cuGraph will handle the communication after workers are assigned.

#. Build the LSF file for Dask work/scheduler
    - The LSF file will use 1 jsrun for scheduler, 1 jsrun for workers.
    - The executable will be launched on batch node, and the workload will be passed onto workers by using Dask's function call.

#. bsub test.lsf


Example test.py for Pagerank:
::

 test.py

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
        sched_file = str(sys.argv[1])  # scheduler file information
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
        ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize, \ 
        delimiter=' ', names=['src', 'dst'], dtype=['int32', 'int32'])
        dg = cugraph.DiGraph()
        dg.from_dask_cudf_edgelist(ddf, 'src', 'dst')
        pr_df = dcg.pagerank(dg, tol=1e-4)
	
        # 4. Shutting down the dask-cuda-cluster
        print("Shutting down the cluster")
        workers_list = list(workers_info)
        disconnect (client, workers_list)
 

Example test.lsf for Launch Dask worker and Dask Scheduler:
::

 Batch.LSF

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


Important Notes on Multi GPUs with cuGraph:
    - Previous example launching test.py with batch node. 
    - I would recommend not to lunch test.py with batch node with cuGraph. CuGraph has a function "batch" that duplicates graph on to all the workers and perform embarrassingly parallel process. The master node also perform the work, hence it is batter avoiding allocation on batch node.
    - Suggested modification to previous LSF file with 2 nodes:

    ::

        # specify total number of workers instead of worker per node. 
        # Leave 1 gpu for the main function for test.py
             jsrun -n 11 -a 1 -c 2 -g 1 --smpiargs="-disable_gpu_hooks" dask-cuda-worker --nthreads 1 \
             --memory-limit 82GB --device-memory-limit 16GB --rmm-pool-size=15GB \
             --death-timeout 60  --interface ib0 --scheduler-file $SCHEDULER_FILE \
             --local-directory $WORKER_DIR --no-dashboard &
        	
             #Wait for WORKERS
             sleep 10
        	
             # This number will be nnode * -r
             WORKERS=11 
             jsrun -n 1 -a 1 -c 1 -g 1 python PATH_EXE/test.py $SCHEDULER_FILE $WORKERS

For more information about running Rapids on Summit: 

https://github.com/benjha/nvrapids_olcf/blob/branch-0.19/docs/nvidia-rapids.rst

For more information about Rapids/cuGraph Lib:

https://docs.rapids.ai/api/cugraph/legacy/

https://github.com/rapidsai/cugraph/tree/branch-0.18
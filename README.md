This programms is dooing the I/O for deep-learning on large images
usage:
For I/O simulating only:

mpirun -n 20 python ./run_training.py OUT_DIR

With deep-Learning (using 4 GPUs on proc 19)
mpirun -n 20 python run_train_tf.py ./log   --num-gpus 4 --num-threads 12 

In the current confiuration, it will only run with 20 procs
The Procs are dooing the following

0-12: Read data for Training, send to 17
13-16: Read data for Testing, send to 18

17: recv data from 0-12, send to 19
18: recv data from 14-16, send to 19

19: do learing and testing (currently just idle)

can be configured in test_params.py in mpi_params

What else to configure (all in test_params.py)


'iterations': number of iterations
'test_interval':200: how often ist tested.
'test_iter': number of iterations in test

'batch_size': how many crops are read before send
'input_size': size of one crop (or box)

Data is read in the get_crop routines in "batch_iter",
the actuall I/O happens in 

def crop_from_array(self, arr)
in the line 
    crop = arr[y0:y1,x0:x1]

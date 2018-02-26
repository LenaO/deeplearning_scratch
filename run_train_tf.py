import image
import batch_iter as BatchIter
import logging
import os
import sys
import test_params as test
import click
import time
import glob
import keras
from mpi4py import MPI
import imp


def add_logfile(logfile):
    fh = logging.handlers.RotatingFileHandler(logfile)
    fh.setFormatter(logging.Formatter('%(asctime)s %(name)s [%(levelname)s]:%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(fh)



def init_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s %(name)s [%(levelname)s]:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger('pytiff').setLevel(logging.INFO)
    logging.getLogger('brainmap.image.image').setLevel(logging.INFO)


def create_split_comm(params, split, comm):
    if params == {} or params['%s_receiver_rank'%split] is None:
        return False
    group = comm.Get_group()
    split_group = group.Incl(params['trainer_ranks'] + params['%s_producer_ranks'%split] + [params['%s_receiver_rank'%split], ])
    split_comm = comm.Create(split_group)

    trainer_ranks = range(len(params['trainer_ranks']))
    producer_ranks = range(len(trainer_ranks),len(trainer_ranks)+len(params['%s_producer_ranks'%split]))
    receiver_rank = len(trainer_ranks) + len(producer_ranks)
    return split_comm, (trainer_ranks, producer_ranks, receiver_rank)

def wrap_batch_iter(split_comm_res, split,  rank, batch_shape, caching, io_caching):
    #NOTE this is a blocking call for receiver_rank and producer_ranks! No common MPI operations will be possible afterwards!
    # only trainer ranks will return!
    #returns batch_iter for each trainer rank
    new_comm, (trainer_ranks, producer_ranks, receiver_rank) = split_comm_res
    # set up producer
    if rank in test.mpi_params['%s_producer_ranks'%split]:
        i = producer_ranks.index(new_comm.Get_rank())
        if caching:
            num_producer_ranks = len(producer_ranks)-1
        else:
            num_producer_ranks = len(producer_ranks)
        batch_iter = eval('BatchIter.get_%s_batch_iter(num_processes=%d, rank=%d)'%(split, num_producer_ranks, i))
        batch_prod = BatchIter.MPIBatchProducer(batch_iter, receiver_rank, comm=new_comm, caching=caching, io_cachpath=io_caching)
        batch_prod.produce()  # NOTE blocking call!
    # set up receiver
    if rank == test.mpi_params['%s_receiver_rank'%split]:
        batch_recv =BatchIter. MPIBatchReceiver(batch_shape, producer_ranks, trainer_ranks, new_comm, caching=caching)
        batch_recv.receive()  # NOTE blocking call!
    if rank in test.mpi_params['trainer_ranks']:
        return BatchIter.MPIBatchIterator(batch_shape, receiver_rank, new_comm)


def get_batch_shape(mode='train'):
    log = logging.getLogger(__name__)
    # batch size 
    batch_size = test.data_params['batch_size']
    if mode == 'test':
        batch_size = test.test_params.get('batch_size', batch_size)
    # number of channels
    n_channels = [len(ch) for ch in test.data_params['input_channels']]
    # size of crops
    sizes = test.data_params['input_size']
    if mode == 'test':
        sizes = test.test_params.get('input_size', sizes)
    batch_shape = [(batch_size, n, size, size) for size, n in zip(sizes, n_channels)]
    log.debug('Batch shape {}'.format(batch_shape))
    return batch_shape



@click.command()
@click.argument('experiment-folder', default='./')
@click.option('--resume-from', default=None, help='path to model checkpoint from which to start training again')
@click.option('--num-gpus', default=1, help='number of gpus to distribute the network over')
@click.option('--num-threads', default=12, help='number of threads to use for training process')
@click.option('--caching/--no-caching', default=False)
@click.option('--io_caching', default=None)
#@mpitools.mpi_profile('profile.out')

def start_training(experiment_folder, resume_from, num_gpus, num_threads, caching, io_caching):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')

    if 'snapshot_dir' not in test.train_params:
        test.train_params['snapshot_dir'] = os.path.join(experiment_folder, 'models')
    if 'log_dir' not in test.train_params:
        test.train_params['log_dir'] = os.path.join(experiment_folder, 'logs')
    batch_shape = get_batch_shape()
    mpi_params = getattr(test, 'mpi_params', {})
    print (mpi_params)
    # add logfiles
    if not mpi_params == {}:
        if rank in mpi_params['trainer_ranks']:
            add_logfile(os.path.join(experiment_folder, '{}.log'.format(timestamp)))
        else:
            add_logfile(os.path.join(experiment_folder, '{}_batch_iter.log'.format(timestamp)))
    else:
        add_logfile(os.path.join(test.train_params['log_dir'], '{}.log'.format(timestamp)))

    batch_iters = []
    split_comm_res_vec = [create_split_comm(mpi_params, 'train', comm), create_split_comm(mpi_params, 'test', comm)]
    for split, split_comm_res in zip(['train', 'test'], split_comm_res_vec):
        if split_comm_res is False:
            # create batch_iter directly
            batch_iters.append(eval('batch_iter.get_%s_batch_iter()'%split))
        else:
            batch_iters.append(wrap_batch_iter(split_comm_res, split, rank, batch_shape, caching, io_caching))

    if mpi_params == {} or rank in mpi_params['trainer_ranks']:

        import tensorflow as tf
        from keras import backend as K
        i = mpi_params.get('trainer_ranks', [0]).index(rank)
        net_def = imp.load_source('net_definition', os.path.join(test.current_dirname ,test.net_definition[i]))

        # Prepare session and graph for training
        tf.reset_default_graph()
        tf_config = tf.ConfigProto(log_device_placement=False, inter_op_parallelism_threads=num_threads)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        K.set_session(sess)
        K.manual_variable_initialization(True)

        if num_gpus > 1:
            # create net on cpu
            with tf.device('/cpu:0'):
                network = net_def.build_net()
            # create parallel network on gpus
            network.make_parallel(num_gpus)
        else:
            network = net_def.build_net()
        from tf_trainer import TFTrainer
        trainer = TFTrainer(network, test.train_params, batch_iters[0], batch_iters[1], timestamp=timestamp)
        trainer.sess = sess
#        # --- TRAINING STARTS ---
        if resume_from == 'LAST':
#            # choose last snapshot from models dir
            resume_from = sorted(glob.glob(os.path.join(experiment_folder, 'models/iter_*.checkpoint.index')))
            if len(resume_from) == 0:
                resume_from = None
            else:
                resume_from = resume_from[-1]
                resume_from = resume_from.replace('.index', '')
        trainer.train()
#        # --- TRAINING ENDS ---
        for batch_iter in batch_iters:
            batch_iter.stop()




if __name__ == '__main__':
    init_logging()

    start_training()


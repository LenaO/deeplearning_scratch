from mpi4py import MPI
default_train_params = {
    # --- General Options ---
    'iterations':100000,
}

class SimpleTrainer(object):
    def __init__(self, params, train_batch_iter, test_batch_iter, timestamp=''):

        self.train_batch_iter = train_batch_iter
        self.test_batch_iter = test_batch_iter
        # set params
        for name in default_train_params.keys():
            setattr(self, name, params.get(name, default_train_params[name]))

    def train(self):
        counter = 0
        while counter < self.iterations:
            batch = self.train_batch_iter.next()
            print("Get batch {}".format(counter))
            counter +=1


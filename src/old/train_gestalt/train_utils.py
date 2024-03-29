from src.utils.misc import make_cuda
from src.old.train_gestalt.callbacks import *
from typing import List


class Logs():
    value = None

    def __repr__(self):
        return f'{self.value}'

    def __repl__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __copy__(self):
        return self.value

    def __deepcopy__(self, memodict={}):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rfloordiv__(self, other):
        return other // self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __floordiv__(self, other):
        return self.value // other

    def __truediv__(self, other):
        return self.value / other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __int__(self):
        return int(self.value)

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __float__(self):
        return float(self.value)

    def __pow__(self, power, modulo=None):
        return self.value ** power

    def __format__(self, format_spec):
        return format(self.value, format_spec)

class ExpMovingAverage(Logs):
    value = None
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def add(self, *args):
        if self.value is None:
            self.value = args[0]
        else:
            self.value = self.alpha * args[0] + (1 -    self.alpha) * self.value
        return self


class CumulativeAverage(Logs):
    value = None
    n = 0

    def add(self, *args):
        if self.value is None:
            self.value = args[0]

        else:
            self.value = (args[0] + self.n*self.value) / (self.n+1)
        self.n += 1
        return self


def run(data_loader, use_cuda, net, callbacks: List[Callback] = None, optimizer=None, loss_fn=None, iteration_step=None, logs=None, logs_prefix='', **kwargs):
    if logs is None:
        logs = {}
    torch.cuda.empty_cache()

    make_cuda(net, use_cuda)

    callbacks = CallbackList(callbacks)
    callbacks.set_model(net)
    callbacks.set_loss_fn(loss_fn)

    callbacks.on_train_begin()

    tot_iter = 0
    epoch = 0
    logs.update({f'{logs_prefix}tot_iter': 0, f'{logs_prefix}stop': False, f'{logs_prefix}epoch': 0})
    while True:
        callbacks.on_epoch_begin(epoch, logs)
        logs[f'{logs_prefix}epoch'] = epoch
        for batch_index, data in enumerate(data_loader, 0):
            callbacks.on_batch_begin(batch_index, logs)
            iteration_step(data, net, loss_fn, optimizer, use_cuda, logs, logs_prefix, **kwargs)
            logs.update({'stop': False})
            logs[f'{logs_prefix}tot_iter'] += 1
            # logs.update({
            #     'loss': loss.item(),
            #     'tot_iter': tot_iter,
            #     'stop': False})

            callbacks.on_training_step_end(batch_index, logs)
            callbacks.on_batch_end(batch_index, logs)
            if logs[f'stop']:
                break
            tot_iter += 1

        callbacks.on_epoch_end(epoch, logs)
        epoch += 1
        if logs[f'stop']:
            break

    callbacks.on_train_end(logs)
    return net, logs
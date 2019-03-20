import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
class Dropout(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        shape = (32, 128, 128, 128)
        x = xp.random.uniform(-1, 1, shape).astype(xp.float32)
        gy = xp.random.uniform(-1, 1, shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.dropout, (x,), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()

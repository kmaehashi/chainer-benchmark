import numpy

import chainer.functions as F

from benchmarks.functions import UnaryMathFunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'cpu')
@parameterize([('dtype', [numpy.float16, numpy.float32])])
class ErfFunc(UnaryMathFunctionBenchmark):

    def setup(self, dtype):
        self.setup_benchmark(F.erf, shape=(5000, 5000), dtype=dtype)

    def time_forward(self, dtype):
        self.forward()

    def time_backward(self, dtype):
        self.backward()

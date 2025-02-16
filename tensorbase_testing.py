from match.tensorbase import TensorBase
import random
import operator


def make_tensorbases(shape, num_tensorbases = 2):
    res = []
    for _ in range(num_tensorbases):
        t = TensorBase(shape)
        t.randn_(10, 1)
        res.append(t)
    return tuple(res)


if __name__ == "__main__":
    m1, m2 = make_tensorbases((1,))
    print(m2)
    print(m2.permute((1,)))


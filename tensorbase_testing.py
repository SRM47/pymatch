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
    m1, m2 = make_tensorbases((1,1,3))
    m3, m4 = make_tensorbases(())
    print(m1@m3)


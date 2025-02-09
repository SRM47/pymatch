from match.tensorbase import TensorBase
import random
import operator


def make_tensorbases(shape, num_tensorbases):
    res = []
    for _ in range(num_tensorbases):
        t = TensorBase(shape)
        t.fill_(0)
        res.append(t)
    return tuple(res)


if __name__ == "__main__":
    m1, m2 = make_tensorbases((4,4,4), 2)
    m1.fill_(2)
    m2.fill_(3)

    m3, m4 = make_tensorbases((1,4), 2)
    m3.fill_(0.5)
    m4.fill_(2)
    m4.reshape_((4, 1))

    m5, m6 = make_tensorbases((4,), 2)
    m5.fill_(6)
    m6.fill_(7)

    print(m3.relu())


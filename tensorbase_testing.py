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
    t1, t2 = make_tensorbases((), 2)
    t1.fill_(2)
    t3, t4 = make_tensorbases((2,1,4), 2)
    t3.fill_(2)
    print(t3._raw_data)
    print(t3.reshape((8,1))._raw_data)
    print(t3._raw_data)

    
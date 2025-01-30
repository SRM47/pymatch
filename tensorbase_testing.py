from match.tensorbase import TensorBase
import random

def make_tensorbases(shape, num_tensorbases):
    res = []
    for _ in range(num_tensorbases):
        t = TensorBase(shape)
        t.fill_(random.gauss(5, 5))
        res.append(t)
    return tuple(res)

if __name__ == "__main__":
    t1, t2 = make_tensorbases((2,3,1), 2)
    t3, t4 = make_tensorbases((2,1,4), 2)
    print(t1, t3)
    print(t1-t3, t1//t3, t3//t1)

    
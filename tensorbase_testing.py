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
    operators_to_test = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "truediv": operator.truediv,
            "floordiv": operator.floordiv,
        }
    for msg, op in operators_to_test.items():
        print(msg, op(1.47, t1))

    
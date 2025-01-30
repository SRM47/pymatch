from match.tensorbase import TensorBase


if __name__ == "__main__":
    x = TensorBase(())
    x.fill_(1)
    print(x)
    print(x.item())
    y = TensorBase((2,3,4))
    y.fill_(5)
    print(y)
    print(y+x)
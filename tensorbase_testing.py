from match.tensorbase import TensorBase


if __name__ == "__main__":
    x = TensorBase((2,3,4))
    x.fill_(1)
    print(x)
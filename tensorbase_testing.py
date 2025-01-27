from match.tensorbase import TensorBase

if __name__ == "__main__":
    x, y = TensorBase((2,3)), TensorBase((2,2))
    print("x")
    x.fill_(5)
    print("y")
    y.fill_(7)
    print("x+y")
    print(x+y)
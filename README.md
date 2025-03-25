# pymatch

A pure-Python, PyTorch-like automatic differentiation library for education.

**pymatch** is a lightweight, educational library designed to demonstrate the fundamentals of automatic differentiation and neural network operations. It provides a simplified, Python-only implementation of core concepts found in popular deep learning frameworks like PyTorch.

**Key Features:**

* **Automatic Differentiation:** Implements reverse-mode automatic differentiation to compute gradients of arbitrary functions.
* **Tensor Operations:** Supports basic tensor operations like addition, subtraction, multiplication, and exponentiation, and others
* **Educational Focus:** Designed to be easily understandable and modifiable, making it ideal for learning about automatic differentiation.
* **Pure Python:** No external dependencies beyond standard Python libraries, enhancing portability and accessibility.

** Future Development Ideas**

* **Conv2d in C**: Implement the Conv2D layer from `/src/match/nn/conv2d.py` in the `src/match/tensorbase` in C.
* **Pytorch-like Optimizers**: Implement standard optimizers like _SGD_, _RMSProp_, _Adam_, _AdamW_, etc., in a new file `src/match/nn/optimizer.py`.
* **Code Clarity and Cleanliness**: Improve the general readability of the code base.
* **LLMs**: Finish the GPT-2 implementation in `/src/match/nn/transformer.py`.
* **Benchmarking**: Conduct speed and memory analysis on basic tensor operations like matrix multiplication, binary operations, and unary operations.
* **Technical Documentation**: Create documentation for TensorBase C backend that includes information on how to use, navigate, and read the codebase.



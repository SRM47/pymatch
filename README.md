# pymatch

A pure-Python, PyTorch-like automatic differentiation library for education.

**pymatch** is a lightweight, educational library designed to demonstrate the fundamentals of automatic differentiation and neural network operations. It provides a simplified, Python-only implementation of core concepts found in popular deep learning frameworks like PyTorch.

### Key Features

* **Automatic Differentiation:** Implements reverse-mode automatic differentiation to compute gradients of arbitrary functions.
* **Tensor Operations:** Supports basic tensor operations like addition, subtraction, multiplication, and exponentiation, and others
* **Educational Focus:** Designed to be easily understandable and modifiable, making it ideal for learning about automatic differentiation.
* **Pure Python:** No external dependencies beyond standard Python libraries, enhancing portability and accessibility.

## 🏛️ Core Architecture

The library is split into two main parts: a fast C backend for mathematical computations and a Python frontend that adds machine learning capabilities.

---

### The C Backend: `tensorbase`

Think of `tensorbase` as the project's high-performance engine. It's written in C for speed and handles all the fundamental linear algebra and operations on n-dimensional arrays (tensors), much like the core of NumPy.

This backend lives in `src/match/tensorbase/` and consists of two key files:

* `tensorbase.h`: This is the **header file**, which acts as a blueprint. It **declares** all the available math functions (like matrix multiplication, addition, etc.) so other parts of the C code know what's available.
* `tensorbasemodule.c`: This is the **Python wrapper**. It serves as the bridge between Python and C. It uses the function declarations from `tensorbase.h` to expose the C functions to Python, allowing you to call the fast C code directly from your Python scripts.

The flow is: Python calls a function in the `tensorbasemodule.c` wrapper, which in turn executes methods defined in `tensorbase.h`

---

### The Python Frontend: `tensor`

The `tensor` module, located at `src/match/tensor.py`, is the smart user interface built on top of the `tensorbase` engine.

It takes the fast, low-level operations provided by the C backend and adds **automatic differentiation**. This feature automatically keeps track of every computation to build a **computation graph**. The graph is then used to calculate gradients (derivatives) automatically. 

In short, `tensorbase` does the heavy lifting (the math), and `tensor` adds the "smarts" (gradient tracking) needed for machine learning.

---

### The Neural Network Library: nn
The `nn` library is the final layer, built on top of the `tensor` API, providing the tools to build and train machine learning models.

All layers (like Linear, Conv2D, and even a Transformer) inherit from a base `Module` class, which provides the core logic for backpropagation, allowing gradients to be passed backward through the computation graph created by the tensor objects. 

The library also includes common activation functions (like `ReLU`) and loss functions (like `Cross-Entropy` Loss), all built on top of the `tensor` API.

---

### Final Architecture Flow (The PyMatch Abstraction Stack)
The complete flow from low-level C code to high-level neural network layers is as follows:

`tensorbase.h` (C Declarations) → Defines the available C functions.

`tensorbasemodule.c` (C Wrapper) → Exposes the C functions to Python.

`tensorbase` (C Backend Package) → The compiled, high-speed math library available in Python.

`tensor` (Python Frontend) → Uses tensorbase and adds automatic differentiation and computation graphs.

`nn` (Neural Network API) → Uses tensor objects to build network layers and implement backpropagation.

## Usage

### 🚀 Running the Demo

This is a pure Python library, but it requires `pytorch` separate package for data loading (coming soon...).

1.  **Install PyTorch**: The PyMatch demo uses `torch` and `torchvision` to load data to train a neural network. You can install it using pip (or any other package manager):
    ```bash
    pip install torch torchvision
    ```
2.  **Run the Demo**: Navigate to the project's root directory (`pymatch/`) and execute the following command to train a linear classifier on the MNIST dataset:
    ```bash
    python3 demo/demo_linear.py
    ```
    > 📝 **Note**: An CNN image classifier demo is coming soon\!

---

### ✅ Running Tests

To run the unit tests, navigate to root directory and use Python's `unittest` module.

For example, to run the tests in `test_module.py`:

```bash
python3 -m unittest tests.test_module
```

Replace `test_module` with the name of any other test file you wish to run.

###  🛠️ Modifying the `tensorbase` C Backend

The default backend is written in C for performance. If you make any changes to the C source code located in `src/match/tensorbase`, you must recompile the shared library for the changes to take effect in Python.

From the project's root directory, run the following command. This will clean previous builds and then recompile the C extension in place, allowing you to `import tensorbase` with your new changes.

```bash
python3 setup.py clean --all && python3 setup.py build_ext --inplace
```

**Important**: You must run this command every time you modify the C backend code.


## Future Development Ideas

* **Conv2d in C**: Implement the Conv2D layer from `/src/match/nn/conv2d.py` in the `src/match/tensorbase` in C.
* **Pytorch-like Optimizers**: Implement standard optimizers like _SGD_, _RMSProp_, _Adam_, _AdamW_, etc., in a new file `src/match/nn/optimizer.py`.
* **Code Clarity and Cleanliness**: Improve the general readability of the code base.
* **LLMs**: Finish the GPT-2 implementation in `/src/match/nn/transformer.py`.
* **Benchmarking**: Conduct speed and memory analysis on basic tensor operations like matrix multiplication, binary operations, and unary operations.
* **Technical Documentation**: Create documentation for TensorBase C backend that includes information on how to use, navigate, and read the codebase.



# pymatch

A pure-Python, PyTorch-like automatic differentiation library for education.

**pymatch** is a lightweight, educational library designed to demonstrate the fundamentals of automatic differentiation and neural network operations. It provides a simplified, Python-only implementation of core concepts found in popular deep learning frameworks like PyTorch.

### Key Features

* **Automatic Differentiation:** Implements reverse-mode automatic differentiation to compute gradients of arbitrary functions.
* **Tensor Operations:** Supports basic tensor operations like addition, subtraction, multiplication, and exponentiation, and others
* **Educational Focus:** Designed to be easily understandable and modifiable, making it ideal for learning about automatic differentiation.
* **Pure Python:** No external dependencies beyond standard Python libraries, enhancing portability and accessibility.


### 🚀 Running the Demo

This is a pure Python library, but it requires `pytorch` separate package for data loading (coming soon...).

1.  **Install PyTorch**: The PyMatch demo ses PyTorch to load data to train a neural network. You can install it using pip:
    ```bash
    pip install torch
    ```
2.  **Run the Demo**: Navigate to the project's root directory (`pymatch/`) and execute the following command to train a linear classifier on the MNIST dataset:
    ```bash
    python3 demo/demo_linear.py
    ```
    > 📝 **Note**: An CNN image classifier demo is coming soon\!


### ✅ Running Tests

To run the unit tests, navigate to the `tests/` directory and use Python's `unittest` module.

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


### Future Development Ideas

* **Conv2d in C**: Implement the Conv2D layer from `/src/match/nn/conv2d.py` in the `src/match/tensorbase` in C.
* **Pytorch-like Optimizers**: Implement standard optimizers like _SGD_, _RMSProp_, _Adam_, _AdamW_, etc., in a new file `src/match/nn/optimizer.py`.
* **Code Clarity and Cleanliness**: Improve the general readability of the code base.
* **LLMs**: Finish the GPT-2 implementation in `/src/match/nn/transformer.py`.
* **Benchmarking**: Conduct speed and memory analysis on basic tensor operations like matrix multiplication, binary operations, and unary operations.
* **Technical Documentation**: Create documentation for TensorBase C backend that includes information on how to use, navigate, and read the codebase.



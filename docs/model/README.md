# Model Implementations and Tests

## Overview

This part illustrates how we can use forward-model automatic differentiation to build an example product, typically a model that can approximate an arbitrary function.

---

## ✨ Classes

* **Linear Model:** Implementation of 2 parameters to be optimized (w and b).
* **Parallel Linear Model:** Uses parallel `for` loops on batches of data.
* **Neural Model:** A basic 2-layer feed-forward neural network implementation.
* **Cache-Optimized Neural Model:** Uses `std::span` to point to a unique, contiguous, flat memory where the parameters are located.
* **Parallel Neural Model:** Leverages OpenMP threads working on meta-batches, parallelizing mini-batches.

---

## 🏗️ Implementations and Tests

### Linear Model and Parallel Linear Model

The comparison test in ("LinearModel.cpp"):

On 100 datapoints, `batch_size` 20, SGD `lr = 0.01`

```bash
Serial training took: 90722 microseconds
w: 4.99575 | b: -2.01084
Parallel training took: 28838 microseconds
Speed Up is 3.14592
```
Although there is a speed up, the convergence is diminished, to have the same approximated parameter, the linear model took `200` epochs, whereas the parallel model worked for `1000` epochs.

### Linear model and Neural Model

![linear vs neural model](model_comparison.png)




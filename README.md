# Vision Transformer Submission

## Summary

This repository contains the submission by the OpenHACK team
for the Hackathon@School of Innovation, held on December 19,
2024. The team focused on optimizing the linear transformation
function implemented in modules.cpp. The results demonstrate
a 33x speedup on the Attention mechanism and a 51x speedup
on the MLP component compared to the serial implementation.

Additionally, the submission includes a cleaner implementation
of the aforementioned function. However, this implementation
is incomplete and currently results in runtime errors.
Despite this limitation, the team has opted to include this
work in the submission as supplementary material.

## Team Members

- Salvatore Andaloro
- Leonardo Falsarolo
- Sophie Motter
- Giovanni Santini

## Submission

The improved algorithm is located in the submission branch
within the `acc_src/modules.cpp` file, specifically in
the `Linear::operator()(const Tensor&, Tensor&)` function at
line 41. The loops are accelerated using OpenACC pragmas,
with the following key considerations:

1. **Memory Transfer**: Memory must be transferred to and
   from the device before and after the parallel section.
   This is managed using the `copy()` directive. However,
   not all C++ classes are trivially copyable. For instance,
   if a class contains pointers to other classes or arrays,
   those members must also be copied to the device memory.
   To address this, our implementation ensures a deep copy
   by including the inner data of the classes in the `copy()`
   directive. A more elegant implementation is proposed later
   in the document.

2. **Loop Independence**: The loops must be explicitly marked
   as independent, as the compiler does not automatically
   parallelize for loops without such annotations.

3. **Inner Loop Optimization**: The innermost loop is accelerated
   using a reduction operation by utilizing the `reduction(+:cumulate)`
   directive.

3.  **Device-Callable Methods**: Most methods cannot be invoked
   from the device without explicit configuration. To address this,
   our implementation avoids calls to Tensor methods directly.
   A more maintainable solution is discussed in subsequent sections.

The improved algorithm achieves an average forward pass time per
batch of 0.000294755 seconds, with 0.0338206 seconds spent
in Attention and 0.035154 seconds in the MLP module when
tested on the `data_2` dataset. This represents a 33x speedup
for Attention and a 51x speedup for the MLP module compared
to the serial implementation.

To address the challenges outlined in points 1 and 2, a second
implementation is proposed in the master branch with the
following enhancements:

1. **Dynamic Device Memory Management**: Instead of copying
   classes and their internal attributes (e.g., arrays or
   pointers to other structures) using the `copy()` directive
   before running the algorithm, device memory is allocated
   and deallocated dynamically based on the lifetime of the
   object. Synchronization between CPU and GPU data is achieved
   using a new method called `update_host`, which updates all
   relevant memory via #pragma acc update.
2. **GPU-Compatible Methods**: The methods are annotated with
   `#pragma acc routine seq`, enabling them to be invoked directly
   from the GPU. This modification simplifies the process of
   writing and maintaining the algorithms.

Despite these planned improvements, the team was unable to fully
complete this implementation.

## Replicate the results

Checkout the submission branch:

```bash
git checkout submission
```

Load the necessary modules and compile the executable:

```bash
module load nvhpc/24.3 &&
srun -N 1 -p boost_usr_prod \
          -A tra24_hckunitn  \
          --reservation=s_tra_hckunitn \
          -t 00:05:00 \
          --gres=gpu:1 \
          make -f makefile.acc acc_bin/vit.exe
```

setup python venv:

```bash
python3 -m venv .venv &&
source .venv/bin/activate &&
pip install numpy
```

run and profile the code:

```bash
chmod +x run_acc.sh &&
srun -N 1 -p boost_usr_prod \
          -A tra24_hckunitn  \
          --reservation=s_tra_hckunitn \
          -t 00:05:00 \
          --gres=gpu:1 \
          ./run_acc.sh --profile
```

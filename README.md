# Vision Transformer Submission

## Summary

This repository contains the submission of the group OpenHACK
for the Hackathon@School of Innovation taking place in 19/12/2024.
The team focused their efforts on optimizing the linear transformation
function implemented in `modules.cpp`. The results exhibit a speedup of
33x on Attention and 51x on MLP compared to the serial implementation.
Furthermore, this submission contains a cleaner implementation of
the said function, however the implementation is not fully complete
and yields runtime errors. Despte this, the team agreed to include
this work on the submission as additional material.

## Team Members

- Salvatore Andaloro
- Leonardo Falsarolo
- Sophie Motter
- Giovanni Santini

## Submission

The improved algorithm can be found in the `submission` branche,
inside the  [acc_src/modules.cpp](./acc_src/modules.cpp) file
in the function `Linear::operator()(const Tensor&, Tensto&)` line 41.
The loops are accelerated using OpenACC pragmas, we highlight 4 notable
considerations:
1. Memory needs to be copied in and out of the device before and after the
   parallel section. This can be achieved using the `copy()` keyword.
   However, not all C++ lasses are trivially copyable. For example, If
   a class contains a pointer to another class or array, this too would
   need to be copied in the device memory. For this reason, our implementation
   includes the inner data of the classes in the `copy()` keyword to perform
   a deep copy. A more elegant implementation is proposed later.
2. The loops needed to be explicitly marked as independent, otherwise
   the compile would not automatically parallelize the for loops.
3. The most inner loop can be accelerated using a reduction operation,
   levereging the `reduction(+:cumulate)` keyword.
4. Most methods cannot be called from the device unless manually configured.
   This implementation removes the calls to the Tensor's methods. A more
   maintainable solution is explained later.

Overall, the improved algorithm takes an average foreward time per batch 
of 0.000294755 s with 0.0338206 s in Attention and 0.035154 s in MLP
using the `data_2` dataset, with an increase of 33x and 51x respectively
compared to the serial implementation.

To address the problems in `1.` and `2.`, we propose a second implementation
in the `master` branch implementing the following changes:
1. Instead of copying the classes and their inner attributes (such as arrays
   or pointer to other structs) before the algorithm with the `copy()` keyword,
   we allocate and deallocate the device memory based on the lifetime
   of the object. To synchronyze the cpu and gpu data, a new method
   called `update_host` that will update all the memory using `#pragma acc update`.
2. The methods are marked with `#pragma acc routine seq`, making them
   callable from the GPU, making the algorithms easier to write.

However, the team did not mange to complete this last implementation.

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

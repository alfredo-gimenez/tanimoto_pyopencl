#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

# @init_ocl
# initializes OpenCL environment
# and compiles an input kernel file
def init_ocl(clfile):
    # Set up OpenCL device(s)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Compile OpenCL kernel
    f = open(clfile, 'r')
    fstr = "".join(f.readlines())
    prg = cl.Program(ctx, fstr).build()
    f.close()

    return [ctx, queue, prg]

# @and_or_ocl
# takes two OpenCL buffers s1, s2,
# and returns two OpenCL buffers,
# s1 & s2 and s1 | s2
def and_or_ocl(ctx, queue, prg, s1_clmem, s2_clmem, s_len):
    # Create OpenCL output buffers
    mf = cl.mem_flags
    and_clmem = cl.Buffer(ctx, mf.READ_WRITE, size=s_len*np.int32(0).nbytes)
    or_clmem = cl.Buffer(ctx, mf.READ_WRITE, size=s_len*np.int32(0).nbytes)

    # Run 'andfunc'
    global_size = (s_len,)
    prg.and_or_func(queue, global_size, None, 
                    s1_clmem, s2_clmem,  # input
                    and_clmem, or_clmem) # output

    return [and_clmem, or_clmem]

# @reduction_ocl
# takes an OpenCL buffer and its size
# and returns the sum of values in the buffer
def reduction_ocl(ctx, queue, prg, s_clmem, s_len):
    # Create OpenCL output buffer
    mf = cl.mem_flags
    sum_clmem = cl.Buffer(ctx, mf.READ_WRITE, size=np.int32(0).nbytes)

    # Parallel reduction phases (log2(n) phases)
    i = 2
    s_len = s_len/2
    while s_len:
        # OpenCL kernel arguments
        lvl = np.int32(i)
        gsize = (s_len,)

        # Run reduction
        prg.reduction(queue, gsize, None, s_clmem, sum_clmem, lvl)

        # Next level arguments
        i = 2*i
        s_len = s_len/2

    return sum_clmem


# @tanimoto_ocl
# takes two fingerprints, fp1 and fp2
# and returns their tanimoto index
# fp1 and fp2 must be numpy arrays
def tanimoto_ocl(fp1_np, fp2_np):
    ctx, queue, prg = init_ocl("tanimoto.cl")

    # Create OpenCL buffers for each fingerprint
    mf = cl.mem_flags
    fp1_clmem = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fp1_np)
    fp2_clmem = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fp2_np)

    # Get fp1 & fp2 and fp1 | fp2
    s_len = fp1_np.shape[0]
    and_clmem, or_clmem = and_or_ocl(ctx, queue, prg, fp1_clmem, fp2_clmem, s_len)

    # Get the sum of bits 
    and_sum_clmem = reduction_ocl(ctx, queue, prg, and_clmem, s_len)
    or_sum_clmem = reduction_ocl(ctx, queue, prg, or_clmem, s_len)

    # Copy result buffers into numpy arrays
    # and then into floats
    and_sum_np = np.array(np.int32(0))
    or_sum_np = np.array(np.int32(0))
    cl.enqueue_copy(queue, and_sum_np, and_sum_clmem)
    cl.enqueue_copy(queue, or_sum_np, or_sum_clmem)
    and_sumf = np.float32(and_sum_np)
    or_sumf = np.float32(or_sum_np)

    print "Sum of bits in a & b:", and_sum_np
    print "Sum of bits in a | b:", or_sum_np
    print "      tanimoto index:", and_sumf / or_sumf

def main():
    # Generate 2 random binary strings of length strlen
    strlen = 65536
    fp1_np = np.array(np.random.randint(0,2,strlen), dtype=np.int32)
    fp2_np = np.array(np.random.randint(0,2,strlen), dtype=np.int32)

    tanimoto_ocl(fp1_np, fp2_np)

main()



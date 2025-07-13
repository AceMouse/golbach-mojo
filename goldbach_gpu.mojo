from math import sqrt, ceildiv, iota
from sys import argv, num_physical_cores, simdwidthof, has_accelerator
from algorithm import sync_parallelize, vectorize
from os import Atomic
from time import monotonic
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from gpu.id import block_idx, thread_idx, block_dim

alias delta = Int(10)
alias interval_size = Int(1e6)
alias sieve_size = Int(1e6)
alias bool_dtype = DType.uint8
alias int_dtype = DType.uint64
alias sieve_layout = Layout.row_major(sieve_size)
alias interval_layout = Layout.row_major(interval_size)


def Conv(n : String) -> Int:
    if "e" in n or "E" in n:
        return Int(Float64(n))
    else:
        return Int(n)
    
def format_float(f:Float64, decimal_places:Int) -> String:
    sf = String(f)
    res = ""
    idx = 0
    after_dot_idx = 0
    for ch in sf.codepoint_slices():
        idx+=1
        if ch == '.':
            after_dot_idx = idx
            break
        res += ch
    if decimal_places:
        res += '.'
        for i in range(0,len(sf)-after_dot_idx):
            if i >= decimal_places:
                break
            res += sf[after_dot_idx+i]
    return res

alias block_size = 32
alias num_blocks = sieve_size


fn sieve_interval(
    sieve_tensor: LayoutTensor[bool_dtype, sieve_layout, MutableAnyOrigin],
    interval_tensor: LayoutTensor[int_dtype, interval_layout, MutableAnyOrigin],
    A:Int
):
    """Calculate the element-wise sum of two vectors on the GPU."""

    var tid = thread_idx.x
    var bid = block_idx.x
    var bdim = block_dim.x
    
    if A+bid < 2 and tid == 0:
        interval_tensor[A+bid] = 0
        return
        
    if bid >= sieve_size or bid >= interval_size+A: # out of bounds
        return
    p = sieve_tensor[bid]
    if p == 0: # not a prime to sieve
        return
    first = bid * Int(A+bid-1//bid)-A + tid*bid
    for i in range(first, interval_size, bdim*bid):
        #        print(i+A, " is not prime (stride ", bdim*bid, ")")
        interval_tensor[i] = 0

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
        return
    ctx = DeviceContext()

    sieve_host_buffer = ctx.enqueue_create_host_buffer[bool_dtype](sieve_size).enqueue_fill(1)
    sieve_device_buffer = ctx.enqueue_create_buffer[bool_dtype](sieve_size).enqueue_fill(1)
    interval_host_buffer = ctx.enqueue_create_host_buffer[int_dtype](interval_size).enqueue_fill(1)
    interval_device_buffer = ctx.enqueue_create_buffer[int_dtype](interval_size).enqueue_fill(1)



    fro = Int(0)
    to = Int(1000000)
    a = 0
    print_pairs = 0
    print_intervals = 0
    print_sieve = 0
    terminal = 1
    args = argv()
    while a < len(args):
        ao = a
        if args[a] in ["--from", "-f"]:
            a += 1
            fro = Conv(args[a])
            a += 1
        if args[a] in ["--to", "-t"]:
            a += 1
            to = Conv(args[a])
            a += 1
        if args[a] in ["--print_sieve", "-p"]:
            print_sieve = 1
        if args[a] in ["--print_pairs", "-p"]:
            print_pairs = 1
        if args[a] in ["--print_intervals", "-p"]:
            print_intervals = 1
        if args[a] in ["--no-terminal", "-nt"]:
            terminal = 0
        if ao == a:
            a+=1

    t1 = monotonic()
    top = sqrt(to)
    if delta*delta >= to:
        top = delta
    top += 1
    if top > sieve_size:
        print("please increase sieve_size!")
    ctx.synchronize()
    sieve_host_buffer[0] = 0
    sieve_host_buffer[1] = 0
    prime_count = 0
    for i in range(top):
        if sieve_host_buffer[i]:
            prime_count+=1
            for p in range(i+i,top, i):
                sieve_host_buffer[p] = 0

    if print_sieve:
        for i in range(top,-1,-1):
            if sieve_host_buffer[i]:
                print(String(i))
        print(String(prime_count))
    
    sieve_host_buffer.enqueue_copy_to(dst=sieve_device_buffer)
    ctx.synchronize()
    sieve_tensor = LayoutTensor[bool_dtype,sieve_layout](sieve_device_buffer)
    interval_tensor = LayoutTensor[int_dtype,interval_layout](interval_device_buffer)
    ctx.synchronize()
    ctx.enqueue_function[sieve_interval](
        sieve_tensor,
        interval_tensor,
        0,
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()
    interval_host_buffer.enqueue_copy_from(src=interval_device_buffer)
    ctx.synchronize()
    if print_intervals:
        prime_count = 0
        for i in range(interval_size-1,-1,-1):
            if interval_host_buffer[i]:
                prime_count += 1
                print(String(i))
        print(prime_count)


    t2 = monotonic()
    return

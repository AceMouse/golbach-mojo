from math import sqrt, ceildiv, iota
from sys import argv, num_physical_cores, simdwidthof, has_accelerator
from algorithm import sync_parallelize, vectorize
from os import Atomic
from time import monotonic
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from gpu.id import block_idx, thread_idx, block_dim
#from max.kernels.nn.arg_nonzero import arg_nonzero, arg_nonzero_shape
alias delta = Int(10e4)
alias interval_size = Int(10e6)
alias prime_interval_size = delta+interval_size
alias shape_size = 2
alias sieve_size = Int(1e7)
alias bool_dtype = DType.uint8
alias int_dtype = DType.uint64
alias sieve_layout = Layout.row_major(sieve_size)
alias interval_layout = Layout.row_major(interval_size)
alias shape_layout = Layout.row_major(shape_size)


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


fn set_interval(
    interval_tensor: LayoutTensor[int_dtype, interval_layout, MutableAnyOrigin],
):
    var idx = block_idx.x*block_dim.x + thread_idx.x
    if idx >= prime_interval_size: # out of bounds
        return
    interval_tensor[idx] = 1
fn sieve_interval(
    sieve_tensor: LayoutTensor[bool_dtype, sieve_layout, MutableAnyOrigin],
    interval_tensor: LayoutTensor[int_dtype, interval_layout, MutableAnyOrigin],
    A:Int
):
    var tid = thread_idx.x
    var bid = block_idx.x*2|1
    var bdim = block_size
    if bid >= sieve_size or bid >= prime_interval_size+A: # out of bounds
        return
    p = sieve_tensor[bid]
    if p == 0: # bid is not prime
        return
    first = bid * ceildiv(A,bid)
    if first < 2*bid:
        first = 2*bid
    first = first - A + tid*bid
#    if tid == 0:
#        print(tid,bid,first)
    for i in range(first, prime_interval_size, block_size*bid):
        #        print(i+A, " is not prime (stride ", bdim*bid, ")")
        interval_tensor[i] = 0

fn primes_from_sieved_interval(
    interval_tensor: LayoutTensor[int_dtype, interval_layout, MutableAnyOrigin],
    A:Int
):
    var idx = block_idx.x*block_dim.x + thread_idx.x
    if idx >= prime_interval_size: # out of bounds
        return
    interval_tensor[idx] *= idx+A

fn check_goldbach(
    sieve_tensor: LayoutTensor[bool_dtype, sieve_layout, MutableAnyOrigin],
    interval_tensor: LayoutTensor[int_dtype, interval_layout, MutableAnyOrigin],
    A_prime:Int,
    A:Int
):
    var idx = block_idx.x*block_dim.x + thread_idx.x
    var n = A+idx*2
    
    if n > A+interval_size or n < 6:
        return

    for lo in range(3,delta,2):
        hi = n-lo
        if hi < A_prime:
            break
        if sieve_tensor[lo] and interval_tensor[hi-A_prime]:
            #            print(lo,"+",hi,"=",n)
            return
    print(n," is a counter example")
    

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
        return
    ctx = DeviceContext()

    sieve_host_buffer = ctx.enqueue_create_host_buffer[bool_dtype](sieve_size).enqueue_fill(1)
    sieve_device_buffer = ctx.enqueue_create_buffer[bool_dtype](sieve_size).enqueue_fill(1)



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
    
    t2 = monotonic()
    sieve_host_buffer.enqueue_copy_to(dst=sieve_device_buffer)
    sieve_tensor = LayoutTensor[bool_dtype,sieve_layout](sieve_device_buffer)
    interval_host_buffer = ctx.enqueue_create_host_buffer[int_dtype](prime_interval_size).enqueue_fill(1)
    interval_device_buffer = ctx.enqueue_create_buffer[int_dtype](prime_interval_size)
    interval_tensor = LayoutTensor[int_dtype,interval_layout](interval_device_buffer)

    nonzero_shape_device_buffer = ctx.enqueue_create_buffer[int_dtype](shape_size)
    nonzero_shape_tensor = LayoutTensor[int_dtype,shape_layout](nonzero_shape_device_buffer)
    nonzero_shape_host_buffer = ctx.enqueue_create_host_buffer[int_dtype](shape_size).enqueue_fill(0)

    prime_device_buffer = ctx.enqueue_create_buffer[int_dtype](prime_interval_size)
    prime_tensor = LayoutTensor[int_dtype,interval_layout](interval_device_buffer)

    for A in range(0,to,interval_size):
        sub = delta if A>=delta else 0
        prime_A = A-sub
        ctx.enqueue_function[set_interval](
            interval_tensor,
            grid_dim=Int(ceildiv(prime_interval_size,block_size)),
            block_dim=block_size,
        )
        ctx.enqueue_function[sieve_interval](
            sieve_tensor,
            interval_tensor,
            prime_A,
            grid_dim=Int(ceildiv(top,2)),
            block_dim=block_size,
        )
        ctx.enqueue_function[check_goldbach](
            sieve_tensor,
            interval_tensor,
            prime_A,
            A,
            grid_dim=Int(ceildiv(prime_interval_size,block_size*2)),
            block_dim = block_size,
        )
        # interval_host_buffer.enqueue_copy_from(src=interval_device_buffer)
        # nonzero_shape_host_buffer.enqueue_copy_from(src=nonzero_shape_device_buffer)
        # ctx.synchronize()
        # if print_intervals:
        #     prime_count = 0
        #     for i in range(prime_interval_size-1,sub-1,-1):
        #         if interval_host_buffer[i]:
        #             prime_count += 1
#       #              print(String(i+prime_A))
        #     print("primes in range [",A,":",A+interval_size,"]:",prime_count)
        #     print("nonzero shape in range [",A,":",A+interval_size,"]:",nonzero_shape_host_buffer[0],nonzero_shape_host_buffer[1])

    ctx.synchronize()

    t3 = monotonic()
    mil10_nums_per_sec = (1e2*Float64(top))/Float64(t2-t1)
    print("Initial sieving took " + String(Float64(t2-t1)/1e9) +" seconds")
    print("Sieved : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s                   ")

    mil10_nums_per_sec = (1e2*Float64(to-fro))/Float64(t3-t2)
    print("Checking took " + String(Float64(t3-t2)/1e9) +" seconds")
    print("Checked : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s                   ")
    return

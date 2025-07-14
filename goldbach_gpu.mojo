# A goldbach conjecture checker written for GPUs
#
# check out this paper for a good introduction https://www.ams.org/journals/mcom/2014-83-288/S0025-5718-2013-02787-1/S0025-5718-2013-02787-1.pdf
# 
#
# We want to check that all even numbers 4 < n < to can be written as lo+hi=n where both lo and hi are prime. 
# 
# There may be multiple valid ways of decomposing n: we are interested in the one with the lowest lo. 
#
# Strategy:
# Sieve of Eratosthenes up to max(sqrt(to), delta), call this the small sieve. (we do this on host as it is super quick)
# loop over the segments [(A,A+interval_size) for A in range(to, interval_size)]:
#   sieve the segment and the delta preceding numbers using the primes in the small sieve which are guaranteed to be enough 
#   for every even n in the segment try decompositions in [(lo,n-lo) for lo in range(3,delta,2)] until both lo and n-lo=hi are prime according to the small and segment sieves.
#   optionally collect stats (really slow right now. >90% of runtime.)


from math import sqrt, ceildiv, iota
from sys import argv, num_physical_cores, simdwidthof, has_accelerator
from algorithm import sync_parallelize, vectorize
from os import Atomic
from time import monotonic
from gpu.host import DeviceContext
from gpu import warp, block
from layout import Layout, LayoutTensor
from gpu.id import block_idx, thread_idx, block_dim
from nn import argmaxmin_gpu
from buffer import NDBuffer

alias delta = Int(1e4)  
alias interval_size = Int(10e6)
alias prime_interval_size = delta+interval_size
alias sieve_size = Int(1e7)
alias bool_dtype = DType.uint8
alias int_dtype = DType.int64
alias sieve_layout = Layout.row_major(sieve_size)
alias interval_layout = Layout.row_major(interval_size)
alias stats_layout = Layout.row_major(delta)
alias max_layout = Layout.row_major(1)

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
alias max_block_size = 1024

fn set_interval(
    interval_tensor: LayoutTensor[mut=True, int_dtype, interval_layout],
):
    var idx = block_idx.x*block_dim.x + thread_idx.x
    if idx >= prime_interval_size: 
        return
    interval_tensor[idx] = 1

fn sieve_interval(
    sieve_tensor: LayoutTensor[mut=False, bool_dtype, sieve_layout],
    interval_tensor: LayoutTensor[mut=True, int_dtype, interval_layout],
    A:Int
):
    var tid = thread_idx.x
    var bid = block_idx.x*2|1 # the odd number we are checking if is prime. 
    var bdim = block_size
    if bid >= sieve_size or bid >= prime_interval_size+A:
        return
    p = sieve_tensor[bid]
    if p == 0: # bid is not prime
        return
    first = bid * ceildiv(A,bid)
    if first < 2*bid:
        first = 2*bid
    first = first - A + tid*bid
    stride = block_size*bid
    for i in range(first, prime_interval_size, stride): # if bid is prime mark multiples as not prime. 
        interval_tensor[i] = 0

fn check_goldbach(
    sieve_tensor: LayoutTensor[mut=False,bool_dtype, sieve_layout],
    interval_tensor: LayoutTensor[mut=False,int_dtype, interval_layout],
    A_prime:Int,
    A:Int
):
    var idx = block_idx.x*block_dim.x + thread_idx.x
    var n = A+idx*2 # the even number we are searching for lo+hi=n decomposition for. 
    
    if n > A+interval_size or n < 6:
        return

    for lo in range(3,delta,2):
        hi = n-lo
        if hi < A_prime:
            break
        if sieve_tensor[lo] and interval_tensor[hi-A_prime]>0:
            if interval_tensor[idx] > 0: # reuse interval tensor as output of lo values 
                interval_tensor[idx] = lo
            else :
                interval_tensor[idx] = -lo
            #print(lo,"+",hi,"=",n)
            return
    print(n," is a counter example")

fn abs_interval(
    interval_tensor: LayoutTensor[mut=True,int_dtype, interval_layout]
):
    var idx = block_idx.x*block_dim.x + thread_idx.x
    if idx >= interval_size:
        return
    interval_tensor[idx] = abs(interval_tensor[idx])

fn max_interval(
    interval_tensor: LayoutTensor[mut=False,int_dtype, interval_layout],
    max_tensor: LayoutTensor[mut=True,int_dtype, max_layout],
):
    var max : Int = 0
    for i in range(thread_idx.x,interval_size,block_dim.x):
        a = interval_tensor[i]
        if a > max:
            max = Int(a)
    max = Int(block.max[dtype=int_dtype,width=1,block_size=max_block_size](max))
    if thread_idx.x == 0:
        max_tensor[0] = Int(max)


fn gather_stats(
    interval_tensor: LayoutTensor[mut=False,int_dtype, interval_layout],
    stats_tensor: LayoutTensor[mut=True,int_dtype, stats_layout],
    max_tensor: LayoutTensor[mut=False,int_dtype, max_layout],
    A:Int
):
    var idx = block_idx.x*block_dim.x + thread_idx.x
    var p = 3+idx*2
    
    if p >= delta or stats_tensor[p] or p > Int(max_tensor[0]):
        return

    for i in range(interval_size):
        lo = interval_tensor[i]
        if lo == p:
            stats_tensor[p] = A+i*2
            return

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
        return

    fro = Int(0)
    to = Int(1000000)
    a = 0
    print_sieve = 0
    print_stats = 0
    args = argv()
    while a < len(args):
        ao = a
        if args[a] in ["--to", "-t"]:
            a += 1
            to = Conv(args[a])
            a += 1
        if args[a] in ["--print_sieve", "-p"]:
            print_sieve = 1
        if args[a] in ["--print_stats", "-p"]:
            print_stats = 1
        if args[a] in ["--help", "-h"]:
            print("usage:", args[0], "[-t|--to <number>][-p|--print_sieve][-p|--print_stats]")
            print("example:", args[0], "-t 1e9 --print_sieve --print_stats")
            return
        if ao == a:
            a+=1
    
    ctx = DeviceContext()

    sieve_host_buffer = ctx.enqueue_create_host_buffer[bool_dtype](sieve_size).enqueue_fill(1)
    sieve_device_buffer = ctx.enqueue_create_buffer[bool_dtype](sieve_size).enqueue_fill(1)

    max_device_buffer = ctx.enqueue_create_buffer[int_dtype](1).enqueue_fill(0)

    stats_host_buffer = ctx.enqueue_create_host_buffer[int_dtype](delta).enqueue_fill(0)
    stats_device_buffer = ctx.enqueue_create_buffer[int_dtype](delta).enqueue_fill(0)

    t1 = monotonic()

    top = sqrt(to)
    if delta > top:
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
                print(i, "is prime")
        print(prime_count, "primes found less than", top)
    
    t2 = monotonic()

    sieve_host_buffer.enqueue_copy_to(dst=sieve_device_buffer)
    sieve_tensor = LayoutTensor[bool_dtype,sieve_layout](sieve_device_buffer)

    interval_host_buffer = ctx.enqueue_create_host_buffer[int_dtype](prime_interval_size).enqueue_fill(1)
    interval_device_buffer = ctx.enqueue_create_buffer[int_dtype](prime_interval_size)
    interval_tensor = LayoutTensor[int_dtype,interval_layout](interval_device_buffer)

    stats_tensor = LayoutTensor[int_dtype,stats_layout](stats_device_buffer)

    max_tensor = LayoutTensor[int_dtype,max_layout](max_device_buffer)
    interval_ndbuffer = NDBuffer[int_dtype, 1, __origin_of(interval_device_buffer)]()
    max_ndbuffer =      NDBuffer[int_dtype, 1, __origin_of(max_device_buffer)]()
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
            grid_dim=Int(ceildiv(top,2)), # a warp for each odd (potentially prime) number
            block_dim=block_size,
        )
        ctx.enqueue_function[check_goldbach](
            sieve_tensor,
            interval_tensor,
            prime_A,
            A,
            grid_dim=Int(ceildiv(interval_size,block_size*2)), # a thread for each even number to check.
            block_dim = block_size,
        )
        if print_stats:
            ctx.enqueue_function[abs_interval](
                interval_tensor,
                grid_dim=Int(ceildiv(prime_interval_size,block_size)),
                block_dim = block_size,
            )
            argmaxmin_gpu.argmaxmin_gpu[
                dtype=int_dtype, output_type=int_dtype, rank=1, largest=True
            ](
                ctx,
                interval_ndbuffer,
                max_ndbuffer,
            )
            #ctx.enqueue_function[max_interval](
            #    interval_tensor,
            #    max_tensor,
            #    grid_dim=1, # use one block because code is bad. We want to use more but still get the correct first occurrence of a lo
            #    block_dim = max_block_size,
            #)
            ctx.enqueue_function[gather_stats](
                interval_tensor,
                stats_tensor,
                max_tensor,
                A,
                grid_dim=Int(ceildiv(delta,block_size)),
                block_dim=block_size
            )
    if print_stats:
        stats_host_buffer.enqueue_copy_from(src=stats_device_buffer)

    ctx.synchronize()
    if print_stats:
        for i in range(delta-1,-1,-1):
            if stats_host_buffer[i]:
                print("first occurrence of lo =",i," is when n =",stats_host_buffer[i])

    t3 = monotonic()
    mil10_nums_per_sec = (1e2*Float64(top))/Float64(t2-t1)
    print("Initial sieving took " + String(Float64(t2-t1)/1e9) +" seconds")
    print("Sieved : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s")

    mil10_nums_per_sec = (1e2*Float64(to-fro))/Float64(t3-t2)
    print("Checking took " + String(Float64(t3-t2)/1e9) +" seconds")
    print("Checked : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s")
    return

# A goldbach conjecture checker written for GPUs or CPUs

check out this paper for a good introduction https://www.ams.org/journals/mcom/2014-83-288/S0025-5718-2013-02787-1/S0025-5718-2013-02787-1.pdf
 

We want to check that all even numbers 4 < n < to can be written as lo+hi=n where both lo and hi are prime. 
 
There may be multiple valid ways of decomposing n: we are interested in the one with the lowest lo. 

## Strategy:
- Sieve of Eratosthenes up to max($\sqrt{to}, delta$), call this the small sieve. (we do this on host as it is super quick)

- loop over the segments `[(A,A+interval_size) for A in range(to, interval_size)]`:

   * sieve the segment and the delta preceding numbers using the primes in the small sieve which are guaranteed to be enough 
   
   * for every even n in the segment try decompositions in `[(lo,n-lo) for lo in range(3,delta,2)]` until both $lo$ and $n-lo=hi$ are prime according to the small and segment sieves.
   
   * optionally collect stats (really slow on GPU right now. >90% of runtime.)

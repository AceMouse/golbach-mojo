from math import sqrt
from math import ceildiv
from sys import argv    
from algorithm import sync_parallelize, vectorize
from collections import BitSet
from os import Atomic
struct Sieve(Movable, StringableRaising):
    var top: Int
    var prime_count: Int
    var data: BitSet[10000000]

    fn __init__(out self):
        self.top = 0;
        self.prime_count = 0
        self.data =  BitSet[10000000]()
    def __str__(self) -> String:
        cols = 20
        out = "top: "+String(self.top)+"\n"
            + "prime count: "+String(self.prime_count)+"\n"
            + "data: \n["
        idx = 0
        while idx < self.top:
            col = 0
            if idx > 0:
                out += "\n"
            while idx < self.top and col < cols:
                out += String(self[idx])
                col += 1
                idx += 1
        return out + "]"

    def __getitem__(self, idx: Int) -> Int:
        return not self.data.test(idx)

    def __setitem__(mut self, idx: Int, value: Int) -> None:
        if value == 0:
            self.data.set(idx)
        else: 
            self.data.clear(idx)


    def expand_sieve(mut self, ntop:Int):
        new_top = ntop + 1
        if new_top <= self.top:
            return
        if self.top  < new_top:
            self.top = new_top
        self[0] = 0
        self[1] = 0
        self.prime_count=0
        for i in range(self.top):
            if self[i]:
                self.prime_count+=1
                for p in range(i+i,self.top, i):
                    self[p] = 0

    def _expand_sieve_interval(mut self, top_of_interval:Int):
        self.expand_sieve(sqrt(top_of_interval))
    
    def is_prime(self, number:Int) -> Int:
        if number <= 0 or number >= self.top or number&1 == 0:
            return number==2
        return self[number]

    def _get_primes_in_range(self,A:Int, B:Int) -> List[Int]:
        low = A
        if low < 0:
            low = 0
        set = BitSet[2000000]()

        if low == 0:
            set.set(0)
            set.set(1)
        if low == 1:
            set.set(0)
        var res : List[Int] = []
        if low <= 2 <= B:
            res.append(2)
        if B <= self.top:
            for p in range(low|1,B,2):
                if self[p]:
                    res.append(p)
            return res
        if low < self.top:
            for p in range(low|1,self.top,2):
                if self[p]:
                    res.append(p)
        sqrt_B = sqrt(B)
        top = self.top
        if sqrt_B < self.top:
            top = sqrt_B
        for p in range(0,top):
            if self[p]:
                first = p*Int((low+p-1)/p)
                first += (first < low)*p
                while first <= B:
                    set.set(first-low)
                    first += p
        for i in range(B-low+1+1):
            if not set.test(i):
                res.append(i+low)
        return res

    def get_primes_in_range(mut self,A:Int, B:Int) -> List[Int]:
        low = A
        if low < 0:
            low = 0
        self._expand_sieve_interval(B)
        set = BitSet[1000000]()
        for i in range(B-low+1+1,-1,-1):
            set.set(i)

        if low == 0:
            set.clear(0)
            set.clear(1)
        if low == 1:
            set.clear(0)
        var res : List[Int] = []
        for p in range(0,self.top):
            if self[p]:
                if A <= p <= B:
                    res.append(p)
                first = p*Int((low+p-1)/p)
                first += (first < low)*p
                while first <= B:
                    set.clear(first-low)
                    first += p
        for i in range(len(set)):
            if set.test(i):
                res.append(i+low)
        return res

struct Mutex:
    var counter:Atomic[DType.int32]
    def __init__(out self):
        self.counter = Atomic[DType.int32](SIMD[DType.int32, 1](0))
    def lock(mut self):
        var zero = SIMD[DType.int32, 1](0)
        while not self.counter.compare_exchange_weak(zero,1):
            pass
    def unlock(mut self):
        self.counter -= 1

struct Stats:
    var firstEncounter:InlineArray[Int,10000]
    var brunEstimate:Float64
    var mutex:Mutex

    def __init__(out self, delta:Int):
        self.mutex = Mutex()
        self.brunEstimate = 0
        self.firstEncounter = InlineArray[Int,10000](fill=0)

    def to_str(self,s:Sieve, top:Int) -> String:
        var max = 0
        for i in range(len(self.firstEncounter)):
            if max < self.firstEncounter[i]:
                max = self.firstEncounter[i]
        l = len(String(max))
        ll = len(String(len(self.firstEncounter)))
        max2 = 0
        out = String("p" +" "*(ll-1) + " | S(n)\n")
        for i in range(len(self.firstEncounter)):
            if (s.is_prime(i) and self.firstEncounter[i] == 0):
                n = String(i)
                out += n+" "*(ll-len(n)) + " | >" + String(top) + "\n"
                break
            if max2 < self.firstEncounter[i]:
                n = String(i)
                out += n+" "*(ll-len(n)) + " | " + String(self.firstEncounter[i]) + "\n"
                max2 = self.firstEncounter[i]

        var min = max 
        var outs:List[String] = []
        for i in range(len(self.firstEncounter)-1,-1,-1):
            if self.firstEncounter[i] != 0 and min >= self.firstEncounter[i]:
                n = String(self.firstEncounter[i])
                outs.append( n+" "*(l-len(n)) + " | " + String(i) + "\n")
                min = self.firstEncounter[i]

        out += "n" +" "*(l-1) + " | p(n)\n"
        for o in reversed(outs):
            out += o
        out += "brun estimate: " + String(self.brunEstimate)
        return out

    def update(mut self, encounters:InlineArray[Int,10000], brun: Float64):
        self.mutex.lock()
        self.brunEstimate += brun
        for i in range(len(encounters)):
            if 0 < encounters[i] < self.firstEncounter[i] or self.firstEncounter[i] == 0:
                self.firstEncounter[i] = encounters[i]
        self.mutex.unlock()


def check_interval(mut s : Sieve, A:Int, B:Int, delta:Int, print_pairs:Int):
    s.expand_sieve(delta)
    primes = s.get_primes_in_range(A-delta,B)
    top = B-(B&1)
    bot = A+(A&1)
    if bot < 4:
        bot = 4
    start_prime_idx = len(primes)-1
    for i in range(top,bot-1,-2):
        while primes[start_prime_idx] >= i-1 and start_prime_idx > 0:
            start_prime_idx-=1
        hi_idx = start_prime_idx
        not_found = 1
        while hi_idx >= 0:
            hi = primes[hi_idx]
            lo = i - hi
            hi_idx -= 1
            if s.is_prime(lo):
                not_found = 0
                if print_pairs:
                    print(String(i) +"=" +String(lo)+"+"+String(hi))
                break
        if not_found:
            print(String(i) +" is a counter example!")
            break

def _check_interval(s : Sieve, A:Int, B:Int, delta:Int, print_pairs:Int, mut stats:Stats):
    var brun : Float64 = 0
    primes = s._get_primes_in_range(A-delta,B)
    prev_i = 0
    while primes[prev_i+1] <= A:
        prev_i += 1
    prev = primes[prev_i]
    for p in primes[prev_i+1:]:
        if p-prev == 2:
            brun += 1/p + 1/prev
        prev = p
    #for p in primes:
        #print(String(p) + ", ")
    top = B-(B&1)
    bot = A+(A&1)
    if bot < 4:
        bot = 4
    start_prime_idx = len(primes)-1
    firstEncounter = InlineArray[Int, 10000](fill=0)
    while primes[start_prime_idx] > top-3 and start_prime_idx > 0:
        start_prime_idx-=1
    for i in range(top,bot-1,-2):
        start_prime_idx -= primes[start_prime_idx] > i-3
        hi_idx = start_prime_idx
        not_found = 1
        hi = primes[hi_idx]
        lo = i - hi
        while hi_idx >= 0:
            hi = primes[hi_idx]
            lo = i - hi
            hi_idx -= 1
            if s.is_prime(lo):
                not_found = 0
                if print_pairs:
                    print(String(i) +"=" +String(lo)+"+"+String(hi))
                break
        if not_found:
            print(String(i) +" is a counter example!")
            break
        firstEncounter[lo] = i
    stats.update(firstEncounter, brun)


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
    for ch in sf:
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

from time import monotonic
from gpu.host import DeviceContext
from sys import has_accelerator

def main():
    delta = Int(10e4)
    fro = Int(0)
    to = Int(10e5)
    max_interval_size = Int(1e6)
    a = 0
    print_pairs = 0
    print_intervals = 0
    terminal = 1
    args = argv()
    while a < len(args):
        ao = a
        if args[a] in ["--delta", "-d"]:
            a += 1
            delta = Conv(args[a])
            a += 1
        if args[a] in ["--from", "-f"]:
            a += 1
            fro = Conv(args[a])
            a += 1
        if args[a] in ["--to", "-t"]:
            a += 1
            to = Conv(args[a])
            a += 1
        if args[a] in ["--max_interval_size", "-m"]:
            a += 1
            max_interval_size = Conv(args[a])
            a += 1
        if args[a] in ["--print_pairs", "-p"]:
            print_pairs = 1
        if args[a] in ["--print_intervals", "-p"]:
            print_intervals = 1
        if args[a] in ["--no-terminal", "-nt"]:
            terminal = 0
        if ao == a:
            a+=1

    t1 = monotonic()
    sieve = Sieve()
    if delta*delta < to:
        sieve.expand_sieve(sqrt(to))
    else:
        sieve.expand_sieve(delta)
#    print(String(sieve))
    t2 = monotonic()
    t3 = t2
    p = -1.0
    tasks = ceildiv(to-fro,max_interval_size)
    var counter = Atomic[DType.int32](SIMD[DType.int32, 1](tasks))
    if terminal:
        #        print("\0337", end='')
        print("\033[s", end='')
    
    var stats : Stats = Stats(delta)
    @parameter
    def worker(wid:Int):
        t = to-wid*max_interval_size
        f = to-(wid+1)*max_interval_size
        if f < 0:
            f = 0
        if print_intervals:
            print("Checking interval ["+String(f)+","+String(t)+"] (diff = " +String(t-f) + ")")
        _check_interval(sieve,f,t,delta,print_pairs,stats)
        counter -= 1
        if print_intervals:
            print("Checked interval ["+String(f)+","+String(t)+"] (diff = " +String(t-f) + ")")
        pp = Float64(tasks-(counter.load()[0]))/tasks
        if Int(p*100) < Int(100*pp):
            t3 = monotonic()
            if terminal:
                #        print("\0338",end='')
                print("\033[u", end='')
            p = pp
            est_time = Float64(t3-t2)/p
            rem = (est_time-(t3-t2))/1e9
            s = Int(rem % 60)
            rem /= 60
            m = Int(rem % 60)
            rem /= 60
            h = Int(rem % 24)
            rem /= 24
            d = Int(rem % 365)
            rem /= 365
            y = Int(rem)
            ys = ""
            if y:
                ys = " "*(y<1000)+" "*(y<100)+" "*(y<10)+"{}y ".format(y)
            ds = ""
            if d or y:
                ds = " "*(d<10)+"{}d ".format(d)
            hs = ""
            if h or d or y:
                hs = " "*(h<10)+"{}h ".format(h)
            ms = ""
            if m or h or d or y:
                ms = " "*(m<10)+"{}m ".format(m)
            ss = " "*(s<10)+"{}s".format(s)
            mil_nums_per_sec = (1e3*Float64((to-fro)*p))/Float64(t3-t2)
            print(" " + String(Int(p*100)) +"% done | Time remaining: {}{}{}{}{}   ".format(ys,ds,hs,ms,ss))
            print(" " + " "*len(String(Int(p*100)))+ "       | Checking : " + String(Int(mil_nums_per_sec))+"e+6 numbers/s                   ")

    sync_parallelize[worker](tasks)
    t3 = monotonic()
    stats.mutex.lock()
    print(stats.to_str(sieve, to))
    stats.mutex.unlock()
    mil10_nums_per_sec = (1e2*Float64(sieve.top))/Float64(t2-t1)
    print("Initial sieving took " + String(Float64(t2-t1)/1e9) +" seconds")
    print("Sieved : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s                   ")

    mil10_nums_per_sec = (1e2*Float64(to-fro))/Float64(t3-t2)
    print("Checking took " + String(Float64(t3-t2)/1e9) +" seconds")
    print("Checked : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s                   ")

from math import sqrt
from math import ceildiv
from sys import argv    
from algorithm import sync_parallelize, vectorize
struct Sieve(Movable, StringableRaising):
    var top: Int
    var prime_count: Int
    var data: List[Int]

    fn __init__(out self):
        self.top = 0;
        self.prime_count = 0
        self.data = []
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
        return self.data[idx]

    def __setitem__(mut self, idx: Int, value: Int) -> None:
        self.data[idx] = value

    def expand_sieve(mut self, ntop:Int):
        new_top = ntop + 1
        if new_top <= self.top:
            return
        while self.top <= new_top:
            self.data.append(1)
            self.top += 1
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
        set = [1]*(B-low+1)
        if low == 0:
            set[0] = 0
            set[1] = 0
        if low == 1:
            set[0] = 0
        var res : List[Int] = []
        for p in range(0,self.top):
            if self[p]:
                if A <= p <= B:
                    res.append(p)
                first = p*Int((low+p-1)/p)
                first += (first < low)*p
                while first <= B:
                    set[first-low] = 0
                    first += p
        for i in range(len(set)):
            if set[i]:
                res.append(i+low)
        return res

    def get_primes_in_range(mut self,A:Int, B:Int) -> List[Int]:
        low = A
        if low < 0:
            low = 0
        self._expand_sieve_interval(B)
        set = [1]*(B-low+1)
        if low == 0:
            set[0] = 0
            set[1] = 0
        if low == 1:
            set[0] = 0
        var res : List[Int] = []
        for p in range(0,self.top):
            if self[p]:
                if A <= p <= B:
                    res.append(p)
                first = p*Int((low+p-1)/p)
                first += (first < low)*p
                while first <= B:
                    set[first-low] = 0
                    first += p
        for i in range(len(set)):
            if set[i]:
                res.append(i+low)
        return res

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
def _check_interval(s : Sieve, A:Int, B:Int, delta:Int, print_pairs:Int):
    primes = s._get_primes_in_range(A-delta,B)
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
#    terminal = 1
    args = argv()
    while a < len(args):
        if args[a] in ["--delta", "-d"]:
            a += 1
            delta = Conv(args[a])
            a += 1
        elif args[a] in ["--from", "-f"]:
            a += 1
            fro = Conv(args[a])
            a += 1
        elif args[a] in ["--to", "-t"]:
            a += 1
            to = Conv(args[a])
            a += 1
        elif args[a] in ["--max_interval_size", "-m"]:
            a += 1
            max_interval_size = Conv(args[a])
            a += 1
        elif args[a] in ["--print_pairs", "-p"]:
            a += 1
            print_pairs = 1
        elif args[a] in ["--print_intervals", "-p"]:
            a += 1
            print_intervals = 1
#        elif args[a] in ["--no-terminal", "-nt"]:
#            a += 1
#            terminal = 0
        else:
            a+=1

    t1 = monotonic()
    sieve = Sieve()
    sieve.expand_sieve(sqrt(to))
    t2 = monotonic()
#    t3 = 0
#    p = -1.0
#    if terminal:
#        print("\0337", end='')
    @parameter
    def worker(wid:Int):
        t = to-wid*max_interval_size
        f = fro
        if t-f > max_interval_size:
            f = t-max_interval_size
        _check_interval(sieve,f,t,delta,print_pairs)
        if print_intervals:
            print("Checked interval ["+String(f)+","+String(t)+"]")
#        if Int(p*100) < Int(100*(to-f)/(to-fro)):
#            if terminal:
#                print("\0338",end='')
#            p = ((to-f))/(to-fro)
#            est_time = Float64(t3-t2)/p
#            rem = (est_time-(t3-t2))/1e9
#            s = Int(rem % 60)
#            rem /= 60
#            m = Int(rem % 60)
#            rem /= 60
#            h = Int(rem % 24)
#            rem /= 24
#            d = Int(rem % 365)
#            rem /= 365
#            y = Int(rem)
#            ys = ""
#            if y:
#                ys = " "*(y<1000)+" "*(y<100)+" "*(y<10)+"{}y ".format(y)
#            ds = ""
#            if d or y:
#                ds = " "*(d<10)+"{}d ".format(d)
#            hs = ""
#            if h or d or y:
#                hs = " "*(h<10)+"{}h ".format(h)
#            ms = ""
#            if m or h or d or y:
#                ms = " "*(m<10)+"{}m ".format(m)
#            ss = " "*(s<10)+"{}s".format(s)
#            mil_nums_per_sec = (1e3*Float64(to-f))/Float64(t3-t2)
#            print(" " + String(Int(p*100)) +"% done | Time remaining: {}{}{}{}{}   ".format(ys,ds,hs,ms,ss))
#            print(" " + " "*len(String(Int(p*100)))+ "       | Checking : " + String(Int(mil_nums_per_sec))+"e+6 numbers/s                   ")

#        print("\033[u", end="")
    tasks = ceildiv(to-fro,max_interval_size)
    sync_parallelize[worker](tasks)
    t3 = monotonic()
    mil10_nums_per_sec = (1e2*Float64(sieve.top))/Float64(t2-t1)
    print("Initial sieving took " + String(Float64(t2-t1)/1e9) +" seconds")
    print("Sieved : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s                   ")

    mil10_nums_per_sec = (1e2*Float64(to-fro))/Float64(t3-t2)
    print("Checking took " + String(Float64(t3-t2)/1e9) +" seconds")
    print("Checked : " + format_float(mil10_nums_per_sec,2)+"e+7 numbers/s                   ")

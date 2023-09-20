# entropy
import math
p = 1
j = 5
o = 0
lp = 0
lj = 0
lo = 0
n = p + j + o

if p != 0:
    lp = p*math.log2(p/n)
if j != 0:
    lj = j*math.log2(j/n)
if o != 0:
    lo = o*math.log2(o/n)
    
result = -1*(lp+lo+lj)/n

print(result)

e1 = 1.0
e2 = 0.721
n1 = 5
n2 = 10-n1

wynik = ((e1*n1)/10.0 + (e2*n2)/10.0)
print("---srednia------")
print(wynik)
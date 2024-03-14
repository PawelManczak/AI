# f(x) =  x^2 -2
# f`(x) = 2x

points = [(1, -1), (2,2)]

def squareError(w):
    return sum((((w*x)**2  -2 - y)**2) for x, y in points)
def dSquareError(w):
    return sum( (2 * ((w*x)**2  -2 - y)*x*2) for x, y in points)
def df(x):
    return 2*x - 2
    
eta = 0.01
w = 3

for t in range(100):
    value = squareError(w)
    gradient = dSquareError(w)
    w = w - eta*gradient
    print('iteration {}, w =  {}, f(w) = {}'.format(t,w, value))
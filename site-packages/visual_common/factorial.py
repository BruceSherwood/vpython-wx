# factorial and combin functions needed in statistical computations
# Bruce Sherwood, Carnegie Mellon University, 2000

def factorial(x):
    if x <= 0.:
        if x == 0.: return 1.
        else: raise ValueError('Cannot take factorial of negative number %d' % x)
    fact = 1.
    nn = 2.
    while nn <= x:
        fact = fact*nn
        nn = nn+1.
    if nn != x+1: raise ValueError('Argument of factorial must be an integer, not %0.1f' % x)
    return fact

def combin(x, y):
    # combin(x,y) = factorial(x)/[factorial(y)*factorial(x-y)]
    z = x-y
    num = 1.0
    if y > z:
        y,z = z,y
    nn = int(z+1.)
    while nn <= x:
        num = num*nn
        nn = nn+1.
    if nn != x+1: raise ValueError('Illegal arguments (%d, %d) for combin function' % (x, y))
    return num/factorial(y)

if __name__ == '__main__':
    print('factorial(6) = 6! =', factorial(6))
    print('combin(6,2) = 6!/(2!(6-2)!) =', combin(6,2))


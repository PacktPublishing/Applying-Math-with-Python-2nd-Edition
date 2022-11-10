import sympy

x = sympy.symbols('x')

f = (x**2 - 2*x)*sympy.exp(3 - x)

fp = sympy.simplify(sympy.diff(f)) # (x*(2 - x) + 2*x - 2)*exp(3 - x)
print(fp)

fp2 = -(x**2 - 4*x + 2)*sympy.exp(3 - x)

print(sympy.simplify(fp2 - fp) == 0)  # True


F = sympy.integrate(fp, x)  
print(F) # (x**2 - 2*x)*exp(3 - x)


import numpy as np

# Example 1: Slash Dot Equation: y/(x^2+y^2) = (x+1)/(x^2+y^2)
# Implicit residual: F = y/(r2) - (x+1)/(r2) = (y - (x+1)) / r2
def F_slashdot(x, y, eps=1e-9):
    r2 = x*x + y*y + eps
    return (y - (x + 1.0)) / r2

# Example 2: Quasar Equation: y = x/(x^2 + y^2)
def F_quasar(x, y, eps=1e-9):
    r2 = x*x + y*y + eps
    return y - x / r2

# Example 3a: Simple Star/Particle: x^2 + y^2 = 0
def F_particle(x, y):
    return x*x + y*y  # zero at (0,0)

# Example 3b: "Black Hole" inversion: 1/(x^2+y^2) = 0  -> residual = 1/(r^2)
def F_blackhole(x, y, eps=1e-9):
    r2 = x*x + y*y + eps
    return 1.0 / r2  # never zero; large near origin

# Example 4: Shadow Line: (y-x)*(y+x)=0 then invert one factor -> (x-y)/(x+y)=0
def F_shadowline_mult(x, y):
    return (y - x) * (y + x)

def F_shadowline_div(x, y, eps=1e-9):
    return (x - y) / (x + y + eps)

# Example 5: Phi Equation: x*(x^2 + y^2 - 1) = 0 -> then x/(x^2+y^2-1)=0
def F_phi_mult(x, y):
    return x * (x*x + y*y - 1.0)

def F_phi_div(x, y, eps=1e-9):
    return x / (x*x + y*y - 1.0 + eps)

# Example 6: Underwater Islands:
# y = 4 sin(x) + sin(2.7 y)  (and 2.8 y)
def F_islands(x, y, a=2.7):
    return y - (4.0 * np.sin(x) + np.sin(a * y))



def _eps(val=1e-9):
    return val

# (A) Classics ---------------------------------------------------------------

def F_heart(x, y):
    # (x^2 + y^2 - 1)^3 - x^2 y^3 = 0
    R = (x*x + y*y - 1.0)
    return R*R*R - (x*x)*(y*y*y)

def F_lemniscate_bern(x, y, a=1.0):
    # (x^2 + y^2)^2 - 2 a^2 (x^2 - y^2) = 0
    r2 = x*x + y*y
    return r2*r2 - 2.0*(a*a)*(x*x - y*y)

def F_cassini(x, y, a=1.0, c=1.2):
    # (x^2 + y^2)^2 - 2 c^2 (x^2 - y^2) - (a^4 - c^4) = 0
    r2 = x*x + y*y
    return r2*r2 - 2.0*(c*c)*(x*x - y*y) - (a**4 - c**4)

def F_superellipse(x, y, a=1.2, b=0.8, n=4.0):
    # |x/a|^n + |y/b|^n - 1 = 0
    return np.power(np.abs(x/a), n) + np.power(np.abs(y/b), n) - 1.0

def F_astroid(x, y, a=1.0):
    # (x/a)^(2/3) + (y/a)^(2/3) - 1 = 0
    e = _eps()
    return np.power(np.abs(x/(a+e)), 2.0/3.0) + np.power(np.abs(y/(a+e)), 2.0/3.0) - 1.0

def F_elliptic_curve(x, y):
    # y^2 = x^3 - x  -> y^2 - (x^3 - x) = 0
    return (y*y) - (x*x*x - x)

# (B) Polynomial lemniscates & complex level sets ---------------------------

def F_poly_lemniscate(x, y, roots=((1,0), (-1,0), (0,1)), radius=1.0):
    # Let p(z) = Π (z - r_i). Plot log|p(z)| - log(radius).
    z = x + 1j*y
    p = np.ones_like(z, dtype=np.complex128)
    for (rx, ry) in roots:
        p *= (z - (rx + 1j*ry))
    mag = np.abs(p) + _eps()
    return np.log(mag) - np.log(radius)

# (C) Transcendental / radial -----------------------------------------------

def F_checker_sine(x, y):
    # sin(x) sin(y) = 0
    return np.sin(x) * np.sin(y)

def F_sumdiff_sine(x, y):
    # sin(x) + sin(y) - sin(x+y) = 0
    return np.sin(x) + np.sin(y) - np.sin(x + y)

def F_sinx_over_r(x, y):
    # sin(r)/r = 0  (residual ~ sin(r)/r)
    r = np.sqrt(x*x + y*y) + _eps()
    return np.sin(r) / r

def F_besselJ0(x, y):
    # J0(r) series approx: 1 - r^2/4 + r^4/64 - r^6/2304
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    return (1.0 - 0.25*r2 + (1.0/64.0)*r4 - (1.0/2304.0)*r6)

# (D) Constructive implicit compositions ------------------------------------

def F_circle(x, y, r=1.0, cx=0.0, cy=0.0):
    return ( (x-cx)**2 + (y-cy)**2 - r*r )

def F_line(x, y, a=1.0, b=0.0, c=0.0):
    # a x + b y + c = 0
    return a*x + b*y + c

def F_soft_union(x, y, k=12.0):
    from fuzzygraph import softmin
    A = F_circle(x, y, r=1.0, cx=-0.7, cy=0.0)
    B = F_line(x, y, a=1.0, b=1.0, c=0.1)
    # Import softmin from python library
    from scipy.special import softmin
    return softmin(np.abs(A), np.abs(B), k=k)

def F_soft_intersection(x, y, k=12.0):
    from fuzzygraph import softmax
    A = F_circle(x, y, r=1.2, cx=0.6, cy=0.0)
    B = F_circle(x, y, r=0.8, cx=-0.4, cy=0.0)
    return softmax(np.abs(A), np.abs(B), k=k)

# (E) Multi-well toy ---------------------------------------------------------

def F_multiwell(x, y):
    def well(x, y, cx, cy):
        return ((x-cx)**2 + (y-cy)**2 - 0.5)**2
    phi = well(x,y,-1.0,0.0) + well(x,y,1.0,0.0) + 0.6*well(x,y,0.0,1.0)
    return phi - 0.15


import numpy as np
from fuzzygraph2 import softmin, softmax, sdf_circle, sdf_line
def _eps(val=1e-9): return val
def F_heart(x,y):
    R=(x*x+y*y-1.0)
    return R*R*R - (x*x)*(y*y*y)
def F_lemniscate_bern(x,y,a=1.0):
    r2=x*x+y*y
    return r2*r2 - 2.0*(a*a)*(x*x - y*y)
def F_cassini(x,y,a=1.0,c=1.2):
    r2=x*x+y*y
    return r2*r2 - 2.0*(c*c)*(x*x - y*y) - (a**4 - c**4)
def F_superellipse(x,y,a=1.2,b=0.8,n=4.0):
    return np.power(np.abs(x/a),n)+np.power(np.abs(y/b),n)-1.0
def F_astroid(x,y,a=1.0):
    e=_eps()
    return np.power(np.abs(x/(a+e)),2.0/3.0)+np.power(np.abs(y/(a+e)),2.0/3.0)-1.0
def F_elliptic_curve(x,y):
    return (y*y)-(x*x*x - x)
def F_poly_lemniscate(x,y,roots=((1,0),(-1,0),(0,1)),radius=1.0):
    z=x+1j*y
    p=np.ones_like(z,dtype=np.complex128)
    for (rx,ry) in roots:
        p*=(z-(rx+1j*ry))
    mag=np.abs(p)+_eps()
    return np.log(mag)-np.log(radius)
def F_checker_sine(x,y): return np.sin(x)*np.sin(y)
def F_sumdiff_sine(x,y): return np.sin(x)+np.sin(y)-np.sin(x+y)
def F_sinx_over_r(x,y):
    r=np.sqrt(x*x+y*y)+_eps()
    return np.sin(r)/r
def F_besselJ0(x,y):
    r2=x*x+y*y; r4=r2*r2; r6=r4*r2
    return 1.0-0.25*r2+(1.0/64.0)*r4-(1.0/2304.0)*r6
def F_soft_union(x,y,k=12.0):
    A=sdf_circle(x,y,r=1.0,cx=-0.7,cy=0.0)
    B=sdf_line(x,y,a=1.0,b=1.0,c=0.1)
    return softmin(A,B,k=k)
def F_soft_intersection(x,y,k=12.0):
    A=sdf_circle(x,y,r=1.2,cx=0.6,cy=0.0)
    B=sdf_circle(x,y,r=0.8,cx=-0.4,cy=0.0)
    return softmax(A,B,k=k)
def F_soft_subtract(x,y,k=12.0):
    A=sdf_circle(x,y,r=1.1,cx=0.0,cy=0.0)
    B=sdf_circle(x,y,r=0.7,cx=0.3,cy=0.0)
    return softmax(A,-B,k=k)
def _polar(x,y):
    r=np.sqrt(x*x+y*y); th=np.arctan2(y,x); return r,th
def F_superformula(x,y,m=6,a=1,b=1,n1=0.3,n2=1.7,n3=1.7):
    r,th=_polar(x,y)
    t1=np.power(np.abs((1/a)*np.cos(m*th/4.0)),n2)
    t2=np.power(np.abs((1/b)*np.sin(m*th/4.0)),n3)
    rs=np.power(t1+t2,-1.0/n1)+_eps()
    return r-rs
def F_rose(x,y,k=5):
    r,th=_polar(x,y); return r-np.abs(np.cos(k*th))
def F_lissajous(x,y,ax=3,by=2,delta=np.pi/3):
    return np.sin(ax*x)+np.sin(by*y+delta)
def F_apollonius(x,y,ax=-1.0,ay=0.0,bx=1.0,by=0.0,mu=1.4):
    A=np.sqrt((x-ax)**2+(y-ay)**2)+_eps()
    B=np.sqrt((x-bx)**2+(y-by)**2)+_eps()
    return A/B - mu
def _escape_potential(z,c,iters=100,bailout=64.0):
    w=z.copy(); absw=np.abs(w); mask=absw<bailout
    nu=np.zeros_like(absw,dtype=float)
    for i in range(iters):
        w[mask]=w[mask]*w[mask]+c
        absw=np.abs(w); newmask=absw<bailout
        escaped=mask & (~newmask)
        if escaped.any():
            val=i+1 - np.log2(np.log(absw[escaped]+1e-12))
            nu[escaped]=val
        mask=newmask
        if not mask.any(): break
    return nu
def F_mandelbrot(x,y,iters=100):
    z0=(x+1j*y)*0.0; c=x+1j*y
    return _escape_potential(z0,c,iters=iters)
def F_julia(x,y,cx=-0.8,cy=0.156,iters=100):
    z0=x+1j*y; c=complex(cx,cy)
    return _escape_potential(z0,c,iters=iters)

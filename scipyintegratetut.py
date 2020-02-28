# Summary
# In this tutorial, we will use the examples of the Planck function and the stellar initial mass function 
# (IMF) to illustrate how to integrate numerically, using the trapezoidal approximation and Gaussian 
# quadrature. We will also explore making a custom class, an instance of which is callable in the same 
# way as a function. In addition, we will encounter astropy’s built-in units, and get a first taste of 
# how to convert between them. Finally, we will use LATEX to make our figure axis labels easy to read.

import numpy as np
from scipy import integrate
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu, BlackBody1D
from astropy import units as u, constants as c
import matplotlib.pyplot as plt

# %matplotlib inline

# The Planck function
# The Planck function describes how a black-body radiates energy. We will explore how to find bolometric 
# luminosity using the Planck function in both frequency and wavelength space.

# Let’s say we have a black-body at 5000 Kelvin. We can find out the total intensity (bolometric) from this 
# object, by integrating the Planck function. The simplest way to do this is by approximating the integral 
# using the trapezoid rule. Let’s do this first using the frequency definition of the Planck function.

# We’ll define a photon frequency grid, and evaluate the Planck function at those frequencies. Those will be 
# used to numerically integrate using the trapezoidal rule. By multiplying a numpy array by an astropy unit, 
# we get a Quantity, which is effectively a combination of one or more numbers and a unit.

nu = np.linspace(1., 3000., 1000) * u.THz
bb5000K_nu = blackbody_nu(in_x=nu, temperature=5000. * u.Kelvin)
plt.plot(nu, bb5000K_nu)
plt.xlabel(r'$\nu$, [{0:latex_inline}]'.format(nu.unit))
plt.ylabel(r'$I_{\nu}$, ' + '[{0:latex_inline}]'.format(bb5000K_nu.unit))
plt.title('Planck function in frequency')
plt.show()

# Using LaTeX for axis labels
# Here, we’ve used LaTeX markup to add nice-looking axis labels. To do that, we enclose LaTeX markup text in dollar
# signs, within a string r'\$ ... \$'. The r before the open-quote denotes that the string is “raw,” and backslashes
# are treated literally. This is the suggested format for axis label text that includes markup.

# Now we numerically integrate using the trapezoid rule.

print(np.trapz(x=nu, y=bb5000K_nu).to('erg s-1 cm-2 sr-1'))

# Now we can do something similar, but for a wavelength grid. We want to integrate over an equivalent wavelength 
# range to the frequency range we did earlier. We can transform the maximum frequency into the corresponding 
# (minimum) wavelength by using the .to() method, with the addition of an equivalency.

lam = np.linspace(nu.max().to(u.AA, equivalencies=u.spectral()),
                  nu.min().to(u.AA, equivalencies=u.spectral()), 1000)
bb5000K_lam = blackbody_lambda(in_x=lam, temperature=5000. * u.Kelvin)
plt.plot(lam, bb5000K_lam)
plt.xlim([1.0e3, 5.0e4])
plt.xlabel(r'$\lambda$, [{0:latex_inline}]'.format(lam.unit))
plt.ylabel(r'$I_{\lambda}$, ' + '[{0:latex_inline}]'.format(bb5000K_lam.unit))
plt.title('Planck function in wavelength')
plt.show()

print(np.trapz(x=lam, y=bb5000K_lam).to('erg s-1 cm-2 sr-1'))

# Notice this is within a couple percent of the answer we got in frequency space, despite our bad sampling at small
#  wavelengths!

# Many astropy functions use units and quantities directly. As you gain confidence working with them, consider 
# incorporating them into your regular workflow. Read more here about how to use units.

# How to simulate actual observations
# As of Fall 2017, astropy does not explicitly support constructing synthetic observations of models like black-body 
# curves. The synphot library does allow this. You can use synphot to perform tasks like turning spectra into visual 
# magnitudes by convolving with a filter curve.

# The stellar initial mass function (IMF)
# The stellar initial mass function tells us how many of each mass of stars are formed. In particular, low-mass stars 
# are much more abundant than high-mass stars are. Let’s explore more of the functionality of astropy using this 
# concept.

# People generally think of the IMF as a power-law probability density function. In other words, if you count the stars 
# that have been born recently from a cloud of gas, their distribution of masses will follow the IMF. Let’s write a 
# little class to help us keep track of that:

class PowerLawPDF(object):
    def __init__(self, gamma, B=1.):
        self.gamma = gamma
        self.B = B
    def __call__(self, x):
        return x**self.gamma / self.B

# The __call__ method
# By defining the method __call__, we are telling the Python interpreter that an instance of the class can be called 
# like a function. When called, an instance of this class, takes a single argument, x, but it uses other attributes 
# of the instance, like gamma and B.

# More about classes
# Classes are more advanced data structures, which can help you keep track of functionality within your code that all 
# works together. You can learn more about classes in this tutorial.

# Integrating using Gaussian quadrature
# In this section, we’ll explore a method of numerical integration that does not require having your sampling grid 
# set-up already. scipy.integrate.quad with reference here takes a function and both a lower and upper bound, and 
# our PowerLawPDF class takes care of this just fine.

# Now we can use our new class to normalize our IMF given the mass bounds. This amounts to normalizing a probability 
# density function. We’ll use Gaussian quadrature (quad) to find the integral. quad returns the numerical value of the
#  integral and its uncertainty. We only care about the numerical value, so we’ll pack the uncertainty into _ 
#  (a placeholder variable). We immediately throw the integral into our IMF object and use it for normalizing!

# To read more about generalized packing and unpacking in Python, look at the original proposal, PEP 448, which was
#  accepted in 2015.

salpeter = PowerLawPDF(gamma=-2.35)
salpeter.B, _ = integrate.quad(salpeter, a=0.01, b=100.)

m_grid = np.logspace(-2., 2., 100)
plt.loglog(m_grid, salpeter(m_grid))
plt.xlabel(r'Stellar mass [$M_{\odot}$]')
plt.ylabel('Probability density')
plt.show()

# How many more M stars are there than O stars?
# Let’s compare the number of M dwarf stars (mass less than 60% solar) created by the IMF, to the number of O stars (mass
# more than 15 times solar).

n_m, _ = integrate.quad(salpeter, a=.01, b=.6)
n_o, _ = integrate.quad(salpeter, a=15., b=100.)
print(n_m / n_o)

# There are almost 21000 as many low-mass stars born as there are high-mass stars!

# Where is all the mass?
# Now let’s compute the relative total masses for all O stars and all M stars born. To do this, weight the 
# Salpeter IMF by mass (i.e., add an extra factor of mass to the integral). To do this, we define a new 
# function that takes the old power-law IMF as one of its arguments. Since this argument is unchanged throughout 
# the integral, it is passed into the tuple args within quad. It’s important that there is only one argument that 
# changes over the integral, and that it is the first argument that the function being integrated accepts.

# Mathematically, the integral for the M stars is m^M=∫{.6M,.01M}mIMF(m)dm
# and it amounts to weighting the probability density function (the IMF) by mass. More generally, you find the 
# value of some property ρ that depends on m by calculating
# ρ(m)^M=∫{.6M,.01M} ρ(m)IMF(m)dm

def IMF_m(m, imf):
    return imf(m) * m

m_m, _ = integrate.quad(IMF_m, a=.01, b=.6, args=(salpeter, ))
m_o, _ = integrate.quad(IMF_m, a=15., b=100., args=(salpeter, ))

print(m_m / m_o)

# http://learn.astropy.org/rst-tutorials/units-and-integration.html?highlight=filtertutorials

# Now compare the total luminosity from all O stars to total luminosity from all M stars. 
# This requires a mass-luminosity relation, like this one which you will use as ρ(m):

def lum_m(m, imf):
    t1 = m < .43 and m > .1
    t2 = m < 2 and m > .43
    t3 = m < 20 and m > 2
    t4 = m < 100 and m > 20
    if t1: 
        return (.23)*m**(.23)*imf(m)
    elif t2:
        return m**4*imf(m)
    elif t3:
        return (1.5)*m**(3.5)*imf(m)
    elif t4:
        return 3200*m*imf(m)
## note owing to luminosity function boundaries we can't compute for stars of luminosity less than .1
## so this represents the lower boundary of integration.  To avoid function step discontinuities we should
## integrate piece wise
l_m1, _ = integrate.quad(lum_m, a=.1, b=.429999, args=(salpeter, )) 
l_m2, _ = integrate.quad(lum_m, a=.4300001, b=.6, args=(salpeter, ))##+ integrate.quad(lum_m, a=.4300001, b=.6, args=(salpeter, ))
l_o1, _ = integrate.quad(lum_m, a=15, b=19.99999, args=(salpeter, )) 
l_o2, _ = integrate.quad(lum_m, a=20.00001, b=99.99999, args=(salpeter, ))##+ integrate.quad(lum_m, a=20.00001, b=99.99999, args=(salpeter, ))
l_m = l_m1 + l_m2
l_o = l_o1 + l_o2
print(l_m/l_o)
## low mass stars are far less luminous and in total produce a .14% total luminosity relative to higher mass stars even given their abundance

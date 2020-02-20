# Third-party dependencies
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

# Set up matplotlib and use a nicer set of plot parameters
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
plt.style.use(astropy_mpl_style)
#%matplotlib inline

# In Astropy, the most common object you’ll work with for coordinates is SkyCoord. A SkyCoord can 
# most easily be created directly from angles as shown below.

# In this tutorial we’ll be converting between frames. Let’s start in the ICRS frame (which happens
#  to be the default.)

# For much of this tutorial we’ll work with the Hickson Compact Group 7. We can create an object 
# either by passing the degrees explicitly (using the astropy units library) or by passing in 
# strings. The two coordinates below are equivalent:

hcg7_center = SkyCoord(9.81625*u.deg, 0.88806*u.deg, frame='icrs')  # using degrees directly
print(hcg7_center)

hcg7_center = SkyCoord('0h39m15.9s', '0d53m17.016s', frame='icrs')  # passing in string format
print(hcg7_center)

# We can get the right ascension and declination components of the object directly by accessing those attributes.

print(hcg7_center.ra)
print(hcg7_center.dec)

# Introducing frame transformations
# astropy.coordinates provides many tools to transform between different coordinate systems. For instance, we can 
# use it to transform from ICRS coordinates (in RA and Dec) to Galactic coordinates.

# To understand the code in this section, it may help to read over the overview of the astropy coordinates scheme. 
# The key piece to understand is that all coordinates in Astropy are in particular “frames” and we can transform 
# between a specific SkyCoord object in one frame to another. For example, we can transform our previously-defined
#  center of HCG 7 from ICRS to Galactic coordinates:

hcg7_center = SkyCoord(9.81625*u.deg, 0.88806*u.deg, frame='icrs')

# There are three different ways of transforming coordinates. Each has its pros and cons, but all should give you the
#  same result. The first way to transform to other built-in frames is by specifying those attributes. For instance, 
#  let’s see the location of HCG 7 in Galactic coordinates.

# Transforming coordinates using attributes:

hcg7_center.galactic

# Transforming coordinates using the transform_to() method and other coordinate object
# The above is actually a special “quick-access” form that internally does the same as what’s in the cell below: it uses
#  the `transform_to() 
#  <http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.transform_to>`__ method 
#  to convert from one frame to another. We can pass in an empty coordinate class to specify what coordinate system 
#  to transform into.

from astropy.coordinates import Galactic  # new coordinate baseclass
hcg7_center.transform_to(Galactic())

# Transforming coordinates using the transform_to() method and a string
# Finally, we can transform using the transform_to() method and a string with the name of a built-in coordinate system.

hcg7_center.transform_to('galactic')

# We can transform to many coordinate frames and equinoxes.

# These coordinates are available by default:

# ICRS

# FK5

# FK4

# FK4NoETerms

# Galactic

# Galactocentric

# Supergalactic

# AltAz

# GCRS

# CIRS

# ITRS

# HCRS

# PrecessedGeocentric

# GeocentricTrueEcliptic

# BarycentricTrueEcliptic

# HeliocentricTrueEcliptic

# SkyOffsetFrame

# GalacticLSR

# LSR

# BaseEclipticFrame

# BaseRADecFrame

# Let’s focus on just a few of these. We can try FK5 coordinates next:

hcg7_center_fk5 = hcg7_center.transform_to('fk5')
print(hcg7_center_fk5)

# And, as with the Galactic coordinates, we can acheive the same result by importing the FK5 class 
# from the astropy.coordinates package. This also allows us to change the equinox.

from astropy.coordinates import FK5
hcg7_center_fk5.transform_to(FK5(equinox='J1975'))  # precess to a different equinox

# Instead, we now have access to the l and b attributes:

print(hcg7_center.galactic.l, hcg7_center.galactic.b)

# Transform frames to get to altitude-azimuth (“AltAz”)
# To actually do anything with observability we need to convert to a frame local to an on-earth observer.
# By far the most common choice is horizontal altitude-azimuth coordinates, or “AltAz”. We first need to 
# specify both where and when we want to try to observe.

# We’ll need to import a few more specific modules:

from astropy.coordinates import EarthLocation
from astropy.time import Time

# Let’s first see the sky position at Kitt Peak National Observatory in Arizona.

# Kitt Peak, Arizona
kitt_peak = EarthLocation(lat='31d57.5m', lon='-111d35.8m', height=2096*u.m)

# For known observing sites we can enter the name directly.

kitt_peak = EarthLocation.of_site('Kitt Peak')

# We can see the list of observing sites:

EarthLocation.get_site_names()

# Let’s check the altitude at 1 AM UTC, which is 6 PM AZ mountain time:

observing_time = Time('2010-12-21 1:00')

# Now we use these to create an AltAz frame object. Note that this frame has some other information about the 
# atmosphere, which can be used to correct for atmospheric refraction. Here we leave that alone, because the 
# default is to ignore this effect (by setting the pressure to 0).

from astropy.coordinates import AltAz

aa = AltAz(location=kitt_peak, obstime=observing_time)
print(aa)

# Now we can transform our ICRS SkyCoord to AltAz to get the location in the sky over Kitt Peak at the requested time.

hcg7_center.transform_to(aa)

# To look at just the altitude we can alt attribute:

hcg7_center.transform_to(aa).alt

# Alright, it’s at 55 degrees at 6 PM, but that’s pretty early to be observing. We could try various times one at a time 
# to see if the airmass is at a darker time, but we can do better: let’s try to create an airmass plot.

# this gives a Time object with an *array* of times
delta_hours = np.linspace(0, 6, 100)*u.hour
full_night_times = observing_time + delta_hours
full_night_aa_frames = AltAz(location=kitt_peak, obstime=full_night_times)
full_night_aa_coos = hcg7_center.transform_to(full_night_aa_frames)

plt.plot(delta_hours, full_night_aa_coos.secz)
plt.xlabel('Hours from 6pm AZ time')
plt.ylabel('Airmass [Sec(z)]')
plt.ylim(0.9,3)
plt.tight_layout()
plt.show()

# Great! Looks like the lowest airmass is in another hour or so (7 PM). But that might still be twilight… When should we 
# start observing for proper dark skies? Fortunately, Astropy provides a get_sun function that can be used to check this. 
# Let’s use it to check if we’re in 18-degree twilight or not.

from astropy.coordinates import get_sun

full_night_sun_coos = get_sun(full_night_times).transform_to(full_night_aa_frames)
plt.plot(delta_hours, full_night_sun_coos.alt.deg)
plt.axhline(-18, color='k')
plt.xlabel('Hours from 6pm AZ time')
plt.ylabel('Sun altitude')
plt.tight_layout()
plt.show()

# Looks like it’s just below 18 degrees at 7 PM, so you should be good to go!

# We can also look at the object altitude at the present time and date:

now = Time.now()
hcg7_center = SkyCoord(9.81625*u.deg, 0.88806*u.deg, frame='icrs')
kitt_peak_aa = AltAz(location=kitt_peak, obstime=now)
print(hcg7_center.transform_to(kitt_peak_aa))
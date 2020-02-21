# Each redshift corresponds to an age of the universe, so if you’re plotting some quantity
#  against redshift, it’s often useful show the universe age too. The relationship between
#   the two changes depending the type of cosmology you assume, which is where astropy.cosmology 
#   comes in. In this tutorial we’ll show how to use the tools in astropy.cosmology to 
#   make a plot like this:

# Set up matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

from IPython.display import Image,display
import matplotlib.image as mpimg
#display(Image(filename="/home/christopher/astropywork/ang_dist.png", width=500))
img = mpimg.imread('/home/christopher/astropywork/ang_dist.png')
imgplot = plt.imshow(img)
plt.show()

# We start with a cosmology object. We will make a flat cosmology (which means that the 
# curvature density Ωk=0) with a hubble parameter of 70
#  km/s/Mpc and matter density ΩM=0.3 at redshift 0. The FlatLambdaCDM cosmology then 
#  automatically infers that the dark energy density ΩΛ must =0.7, because ΩM+ΩΛ+Ωk=1.

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# In this case we just need to define the matter density
# and hubble parameter at z=0.

# Note the default units for the hubble parameter H0 are km/s/Mpc.
# We will pass in a `Quantity` object with the units specified.

cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3)

# Note that we could instead use one of the built-in cosmologies, like WMAP9 or Planck13, in 
# which case we would just redefine the cosmo variable.

# Now we need an example quantity to plot versus redshift. Let’s use the angular diameter 
# istance, which is the physical transverse distance (the size of a galaxy, say) corresponding 
# to a fixed angular separation on the sky. To calculate the angular diameter distance for 
# a range of redshifts:

import numpy as np
zvals = np.arange(0, 6, 0.1)
dist = cosmo.angular_diameter_distance(zvals)

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(zvals, dist)
plt.show()

# To check the units of the angular diameter distance, take a look at the unit attribute:

dist.unit

# Now let’s put some age labels on the top axis. We’re going to pick a series of round age values
#  where we want to place axis ticks. You may need to tweak these depending on your redshift range
#   to get nice, evenly spaced ticks.

ages = np.array([13, 10, 8, 6, 5, 4, 3, 2, 1.5, 1.2, 1])*u.Gyr

# To link the redshift and age axes, we have to find the redshift corresponding to each age. The 
# function z_at_value does this for us.

from astropy.cosmology import z_at_value
ageticks = [z_at_value(cosmo.age, age) for age in ages]

# Now we make the second axes, and set the tick positions using these values.

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(zvals, dist)
ax2 = ax.twiny()
ax2.set_xticks(ageticks)
plt.show()

# We have ticks on the top axis at the correct ages, but they’re labelled with the redshift, not the
#  age. We can fix this by setting the tick labels by hand.

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(zvals, dist)
ax2 = ax.twiny()
ax2.set_xticks(ageticks)
ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
plt.show()

# We need to make sure the top and bottom axes have the same redshift limits. They may not line up properly
#  in the above plot, for example, depending on your setup (the age of the universe should be ~13 Gyr at z=0).

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(zvals, dist)
ax2 = ax.twiny()
ax2.set_xticks(ageticks)
ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
zmin, zmax = 0.0, 5.9
ax.set_xlim(zmin, zmax)
ax2.set_xlim(zmin, zmax)
plt.show()

# We’re almost done. We just need to label all the axes, and add some minor ticks. Let’s also tweak the y axis 
# limits to avoid putting labels right near the top of the plot.

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(zvals, dist)
ax2 = ax.twiny()
ax2.set_xticks(ageticks)
ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
zmin, zmax = 0, 5.9
ax.set_xlim(zmin, zmax)
ax2.set_xlim(zmin, zmax)
ax2.set_xlabel('Time since Big Bang (Gyr)')
ax.set_xlabel('Redshift')
ax.set_ylabel('Angular diameter distance (Mpc)')
ax.set_ylim(0, 1890)
ax.minorticks_on()
plt.show()

# Now for comparison, let’s add the angular diameter distance for a different cosmology, from the Planck 2013 results.
# And then finally, we save the figure to a png file.

from astropy.cosmology import Planck13
dist2 = Planck13.angular_diameter_distance(zvals)

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(zvals, dist2, label='Planck 2013')
ax.plot(zvals, dist, label=
        '$h=0.7,\ \Omega_M=0.3,\ \Omega_\Lambda=0.7$')
ax.legend(frameon=0, loc='lower right')
ax2 = ax.twiny()
ax2.set_xticks(ageticks)
ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
zmin, zmax = 0.0, 5.9
ax.set_xlim(zmin, zmax)
ax2.set_xlim(zmin, zmax)
ax2.set_xlabel('Time since Big Bang (Gyr)')
ax.set_xlabel('Redshift')
ax.set_ylabel('Angular diameter distance (Mpc)')
ax.minorticks_on()
ax.set_ylim(0, 1890)
fig.savefig('/home/christopher/astropywork/ang_dist2.png', dpi=200, bbox_inches='tight')
plt.show()

# bbox_inches='tight' automatically trims any whitespace from around the plot edges.

# And we’re done!
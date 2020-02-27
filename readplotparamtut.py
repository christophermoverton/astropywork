# Summary
# This tutorial demonstrates the use of astropy.io.ascii for reading ASCII data, astropy.coordinates
#  and astropy.units for converting RA (as a sexagesimal angle) to decimal degrees, and matplotlib 
#  for making a color-magnitude diagram and on-sky locations in a Mollweide projection.

import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

# Astropy provides functionality for reading in and manipulating tabular data through the astropy.table
#  subpackage. An additional set of tools for reading and writing ASCII data are provided with the
#   astropy.io.ascii subpackage, but fundamentally use the classes and methods implemented in 
#   astropy.table.

# We’ll start by importing the ascii subpackage:

from astropy.io import ascii

# For many cases, it is sufficient to use the ascii.read('filename') function as a black box for reading 
# data from table-formatted text files. By default, this function will try to figure out how your data 
# is formatted/delimited (by default, guess=True). For example, if your data are:

tbl = ascii.read("simple_table.csv")
print(tbl)

# The header names are automatically parsed from the top of the file, and the delimiter is inferred from the 
# rest of the file – awesome! We can access the columns directly from their names as ‘keys’ of the table object:

print(tbl["ra"])

# If we want to then convert the first RA (as a sexagesimal angle) to decimal degrees, for example, we can pluck
#  out the first (0th) item in the column and use the coordinates subpackage to parse the string:

import astropy.coordinates as coord
import astropy.units as u

first_row = tbl[0] # get the first (0th) row
ra = coord.Angle(first_row["ra"], unit=u.hour) # create an Angle object
print(ra.degree) # convert to degrees

# Now let’s look at a case where this breaks, and we have to specify some more options to the read() function.
#  Our data may look a bit messier:

# If we try to just use ascii.read() on this data, it fails to parse the names out and the column names become
#  col followed by the number of the column:

tbl = ascii.read("Young-Objects-Compilation.csv")
print(tbl.colnames)

# What happened? The column names are just col1, col2, etc., the default names if ascii.read() is unable to parse 
# out column names. We know it failed to read the column names, but also notice that the first row of data are 
# strings – something else went wrong!

# A few things are causing problems here. First, there are two header lines in the file and the header lines are 
# not denoted by comment characters. The first line is actually some meta data that we don’t care about, so we want 
# to skip it. We can get around this problem by specifying the header_start keyword to the ascii.read() function. 
# This keyword argument specifies the index of the row in the text file to read the column names from:

tbl = ascii.read("Young-Objects-Compilation.csv", header_start=1)
print(tbl.colnames)

# Great! Now the columns have the correct names, but there is still a problem: all of the columns have string data types, 
# and the column names are still included as a row in the table. This is because by default the data are assumed to start 
# on the second row (index=1). We can specify data_start=2 to tell the reader that the data in this file actually start 
# on the 3rd (index=2) row:

tbl = ascii.read("Young-Objects-Compilation.csv", header_start=1, data_start=2)

# Some of the columns have missing data, for example, some of the RA values are missing (denoted by – when printed):

print(tbl['RA'])

# This is called a Masked column because some missing values are masked out upon display. If we want to use this numeric 
# data, we have to tell astropy what to fill the missing values with. We can do this with the .filled() method. For 
# example, to fill all of the missing values with NaN’s:

print(tbl['RA'].filled(np.nan))

# Let’s recap what we’ve done so far, then make some plots with the data. Our data file has an extra line above the 
# column names, so we use the header_start keyword to tell it to start from line 1 instead of line 0 (remember Python 
# is 0-indexed!). We then used had to specify that the data starts on line 2 using the data_start keyword. Finally, we 
# note some columns have missing values.

data = ascii.read("Young-Objects-Compilation.csv", header_start=1, data_start=2)

# Now that we have our data loaded, let’s plot a color-magnitude diagram.

# Here we simply make a scatter plot of the J-K color on the x-axis against the J magnitude on the y-axis. We use a trick 
# to flip the y-axis plt.ylim(reversed(plt.ylim())). Called with no arguments, plt.ylim() will return a tuple with the axis 
# bounds, e.g. (0,10). Calling the function with arguments will set the limits of the axis, so we simply set the limits to 
# be the reverse of whatever they were before. Using this pylab-style plotting is convenient for making quick plots and 
# interactive use, but is not great if you need more control over your figures.

plt.scatter(data["Jmag"] - data["Kmag"], data["Jmag"]) # plot J-K vs. J
plt.ylim(reversed(plt.ylim())) # flip the y-axis
plt.xlabel("$J-K_s$", fontsize=20)
plt.ylabel("$J$", fontsize=20)
plt.show()

# As a final example, we will plot the angular positions from the catalog on a 2D projection of the sky. Instead of using 
# pylab-style plotting, we’ll take a more object-oriented approach. We’ll start by creating a Figure object and adding a
#  single subplot to the figure. We can specify a projection with the projection keyword; in this example we will use a
#  ollweide projection. Unfortunately, it is highly non-trivial to make the matplotlib projection defined this way follow 
#  the celestial convention of longitude/RA increasing to the left.

# The axis object, ax, knows to expect angular coordinate values. An important fact is that it expects the values to be in 
# radians, and it expects the azimuthal angle values to be between (-180º,180º). This is (currently) not customizable, so
#  we have to coerce our RA data to conform to these rules! astropy provides a coordinate class for handling angular values,
#   astropy.coordinates.Angle. We can convert our column of RA values to radians, and wrap the angle bounds using this class.

ra = coord.Angle(data['RA'].filled(np.nan)*u.degree)
ra = ra.wrap_at(180*u.degree)
dec = coord.Angle(data['Dec'].filled(np.nan)*u.degree)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="mollweide")
ax.scatter(ra.radian, dec.radian)
plt.show()

# By default, matplotlib will add degree tick labels, so let’s change the horizontal (x) tick labels to be in units of hours,
#  and display a grid:

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="mollweide")
ax.scatter(ra.radian, dec.radian)
ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
ax.grid(True)
plt.show()

# We can save this figure as a PDF using the savefig function:

fig.savefig("map.pdf")
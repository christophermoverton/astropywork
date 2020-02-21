# This tutorial describes how to read in and edit a FITS header, and then write it back out to disk. 
# For this example we’re going to change the OBJECT keyword.

from astropy.io import fits

# astropy.io.fits provides a lot of flexibility for reading FITS files and headers, but most of the time 
# the convenience functions are the easiest way to access the data. fits.getdata() reads only the data 
# from a FITS file, but with the header=True keyword argument will also read the header.

data, header = fits.getdata("input_file.fits", header=True)

# There is also a dedicated function for reading only the header:

hdu_number = 0 # HDU means header data unit
fits.getheader('input_file.fits', hdu_number)

# But getdata() can get both the data and the header, so it’s a useful command to remember. Since the primary 
# HDU of a FITS file must contain image data, the data is now stored in a numpy array. The header is stored in
#  an object that acts like a standard Python dictionary.

# But hdu_number = 0 is the PRIMARY HDU.How many HDUs are in this file?
fits_inf = fits.open("input_file.fits")
fits_inf.info()
fits_inf[0].header

# Using fits.open allows us to look more generally at our data. fits_inf[0].header gives us the same output as 
# fits.getheader. What will you learn if you type fits_inf[1].header? Based on fits_inf.info() can you guess what 
# ill happen if you type fits_inf[2].header?

# Now let’s change the header to give it the correct object:

header['OBJECT'] = "M31"

# Finally, we have to write out the FITS file. Again, the convenience function for this is the most useful command to remember:

fits.writeto('output_file.fits', data, header, overwrite=True)

# That’s it; you’re done!

# Two common and more complicated cases are worth mentioning (but if your needs are much more complex, you should 
# consult the full documentation http://docs.astropy.org/en/stable/io/fits/).

# The first complication is that the FITS file you’re examining and editing might have multiple HDU’s (extensions),
#  in which case you can specify the extension like this:

data1, header1 = fits.getdata("input_file.fits", ext=1, header=True)

# This will get you the data and header associated with the index=1 extension in the FITS file. Without specifying a 
# number, getdata() will get the 0th extension (equivalent to saying ext=0).

# Another useful tip is if you want to overwrite an existing FITS file. By default, writeto() won’t let you do this,
#  so you need to explicitly give it permission using the clobber keyword argument:

fits.writeto('output_file.fits', data, header, overwrite=True)

# A final example is if you want to make a small change to a FITS file, like updating a header keyword, but you don’t 
# want to read in and write out the whole file, which can take a while. Instead you can use the mode='update' read mode 
# to do this:

with fits.open('input_file.fits', mode='update') as filehandle:
    filehandle[0].header['MYHDRKW'] = "My Header Keyword"
# Python standard-library
from urllib.parse import urlencode
from urllib.request import urlretrieve

# Third-party dependencies
from astropy import units as u
from astropy.coordinates import SkyCoord
from IPython.display import Image

# Describing on-sky locations with coordinates
# The SkyCoord class in the astropy.coordinates package is used to represent celestial coordinates. 
# First, we’ll make a SkyCoord object based on our object’s name, “Hickson Compact Group 7” or 
# “HCG 7” for short. Most astronomical object names can be found by SESAME, a service which queries
# Simbad, NED, and VizieR and returns the object’s type and its J2000 position. This service can be 
# used via the SkyCoord.from_name() class method:

# initialize a SkyCood object named hcg7_center at the location of HCG 7
hcg7_center = SkyCoord.from_name('HCG 7')

# Note that the above command requires an internet connection. If you don’t have one, execute the 
# following line instead:

# uncomment and run this line if you don't have an internet connection
# hcg7_center = SkyCoord(9.81625*u.deg, 0.88806*u.deg, frame='icrs')

type(hcg7_center)

# Show the available methods and attributes of the SkyCoord object we’ve created called hcg7_center

dir(hcg7_center)

# Show the RA and Dec.

print(hcg7_center.ra, hcg7_center.dec)
print(hcg7_center.ra.hour, hcg7_center.dec)

# We see that, according to SESAME, HCG 7 is located at ra = 9.849 deg and dec = 0.878 deg.

# This object we’ve just created has various useful ways of accessing the information contained within 
# it. In particular, the ra and dec attributes are specialized Quantity objects (actually, a subclass 
# called Angle, which in turn is subclassed by Latitude and Longitude). These objects store angles and 
# provide pretty representations of those angles, as well as some useful attributes to quickly convert 
# to common angle units:

type(hcg7_center.ra), type(hcg7_center.dec)

hcg7_center.ra, hcg7_center.dec

hcg7_center

hcg7_center.ra.hour

# SkyCoord will also accept string-formatted coordinates either as separate strings for RA/Dec or a single
#  string. You’ll need to give units, though, if they aren’t part of the string itself.

SkyCoord('0h39m15.9s', '0d53m17.016s', frame='icrs')

hcg7_center.ra.hour

# Now that we have a SkyCoord object, we can try to use it to access data from the Sloan Digitial Sky Survey 
# (SDSS).
#  Let’s start by trying to get a picture using the SDSS image cutout service to make sure HCG 7 is in the 
#  SDSS footprint and has good image quality.

# This requires an internet connection, but if it fails, don’t worry: the file is included in the repository 
# so you can just let it use the local file'HCG7_SDSS_cutout.jpg', defined at the top of the cell.

# tell the SDSS service how big of a cutout we want
im_size = 12*u.arcmin # get a 12 arcmin square
im_pixels = 1024
cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
query_string = urlencode(dict(ra=hcg7_center.ra.deg,
                              dec=hcg7_center.dec.deg,
                              width=im_pixels, height=im_pixels,
                              scale=im_size.to(u.arcsec).value/im_pixels))
url = cutoutbaseurl + '?' + query_string

# this downloads the image to your disk
urlretrieve(url, 'HCG7_SDSS_cutout.jpg')

Image('HCG7_SDSS_cutout.jpg')
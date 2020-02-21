# astropywork

I use Anaconda 3 installation of Astropy specifically set with environment and python selected interpreter in visual studio code for such environment (Ctrl + Shift + P) to set python interpreter environment.

At bash if you have problems with missing modules as I had, you can use the usual pip install command with an activated anaconda environment.

In my installation I added the line: plt.show() in spectrum.py of the synphot module installation.  Otherwise I had problems with unit conversion problems of the dereddening tutorial...you can populate in visual studio code the reference to this file right clicking on the tutorial (for instance) at line 149.  This pulls up the spectrum.py file, and more specifically under the _do_plot(...) method adding plt.show() after line 733

This is a non interactive notebook version of astropy tutorials.

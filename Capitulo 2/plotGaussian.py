# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Christian Alvarez Pelaez

import pylab as pl
import numpy as np

gaussian = lambda x: 1/(np.sqrt(2*np.pi)*1.5)*np.exp(-(x-0)**2/(2*(1.5**2)))
x = np.arange(-5,5,0.01)
y = gaussian(x)
pl.ion()
pl.plot(x,y,'k',linewidth=3)
pl.xlabel('x')
pl.ylabel('y(x)')
pl.axis([-5,5,0,0.3])
pl.title('Gaussian Function (mean 0, standard deviation 1.5)')
pl.show()
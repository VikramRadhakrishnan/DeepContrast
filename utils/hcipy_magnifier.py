from hcipy import *
from hcipy.optics.optical_element import make_polychromatic

import numpy as np
from matplotlib import pyplot as plt

class MonochromaticMagnifier(OpticalElement):
	'''A monochromatic magnifier for electric fields.
	
	This magnifies the wavefront with a certain magnification factor.
	It does not take into acount propagation effects.
	
	
	Parameters
	----------
	magnification : scalar
		The magnification we want to apply to the grid of the wavefront.
	wavelength : scalar
		The wavelength at which the magnification is defined.
	'''
	def __init__(self, magnification, wavelength=1):
		self.magnification = magnification
		self.wavelength = wavelength
	
	def forward(self, wavefront):

		new_grid = wavefront.electric_field.grid.scaled(self.magnification)

		wf = Wavefront(Field(wavefront.electric_field.copy(), new_grid), wavefront.wavelength)
		wf.total_power = wavefront.total_power

		return wf
	
	def backward(self, wavefront):

		new_grid = wavefront.electric_field.grid.scaled(self.magnification)

		wf = Wavefront(Field(wavefront.electric_field.copy(), new_grid), wavefront.wavelength)
		wf.total_power = wavefront.total_power

		return wf

Magnifier = make_polychromatic(["magnification"])(MonochromaticMagnifier)

def test_chromatic_magnification():
	pupil_grid = make_pupil_grid(128, 1)
	aperture = circular_aperture(1)(pupil_grid)

	mag = Magnifier(magnification=lambda x : 1/100 * (1 + (x-1)))

	wf = Wavefront(aperture, wavelength=1)
	wfm = mag.forward(wf)

	wf2 = Wavefront(aperture, wavelength=0.1)
	wfm2 = mag.forward(wf2)
	
	plt.subplot(2,2,1)
	imshow_field(wf.power)

	plt.subplot(2,2,2)
	imshow_field(wfm.power)
	
	plt.subplot(2,2,3)
	imshow_field(wf2.power)

	plt.subplot(2,2,4)
	imshow_field(wfm2.power)

	plt.show()

def test_achromatic_magnification():
	pupil_grid = make_pupil_grid(128, 1)
	aperture = circular_aperture(1)(pupil_grid)

	mag = Magnifier(1/100)

	wf = Wavefront(aperture, wavelength=1)
	wfm = mag.forward(wf)
	
	plt.subplot(1,2,1)
	imshow_field(wf.power)

	plt.subplot(1,2,2)
	imshow_field(wfm.power)
	
	plt.show()


if __name__ == "__main__":
	test_chromatic_magnification()
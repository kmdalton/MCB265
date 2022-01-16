"""
Library for simulating scattering onto a detector according to Klein-Nishina equation
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.constants import *
from scipy.interpolate import interp1d
from os.path import dirname,realpath




#I know global variables are evil, but these are constants which i really want to keep consistent
mu = electron_mass*c*c/electron_volt
ro = physical_constants['classical electron radius'][0]
ro2= 1e28*(ro**2)

#So tired of reinventing this
def ev2angstrom(energy):
    return 1e10*h*speed_of_light/(energy*electron_volt)

#Hint: this is the same function as above 0.o
def angstrom2ev(wavelength):
    return ev2angstrom(wavelength)

class interpolator(dict):
    def __init__(self, inFN):
        self.inFN = inFN
        data = np.loadtxt(inFN, skiprows=1, delimiter=',')
        keys = map(int, open(inFN).readline().strip().split(',')[1:])
        for k,v in zip(keys,data[:,1:].T):
            self[k] = interp1d(data[:,0], v, kind='quadratic')


dir_path = dirname(realpath(__file__))
scatteringfunction = interpolator(dir_path + '/scatteringfunction.txt')
formfactor         = interpolator(dir_path + '/formfactor.txt')

def transform_cartesian(theta, phi, z):
    """
    Returns
    -------
    theta : float or array
    phi   : float or array
    """
    theta = np.arctan2(np.hypot(x - xpos, y - ypos), -z)
    phi = np.arctan2(y-ypos , x - xpos)
    return theta, phi
    
def transform_spherical(x, y, z):
    """
    transform_spherical(x, y, z, **kw) converts coordinates to the scattering and azimuthal angles, phi and theta. assumes polarization in y plane
    We're using the convention according to A.L. Hanson. The calculation of scattering cross sections for polarized x-rays. Nuclear
    Instruments and Methods in Physics Research (1986) 583-598.  This means that theta is from runs from 0 to pi with 0 referring to 
    precise backscatter and pi meaning precise forward scatter.

    Parameters
    ----------
    x : float or array
    y : float or array
    z : float or array
        This is the distance from the sample to the detector. positive z direction is in the direction of the incident photon wavevector. So,
        negative z values are asking for information about backscatter. 

    Returns
    -------
    theta : float or array
    phi   : float or array
    """
    theta = np.arctan2(np.hypot(x, y), -z)
    phi = np.arctan2(y, x)
    return theta, phi

def compton(theta, phi, ko, Z=None):
    """
    compute the compton portion of the differential intensity of scattering for a given theta and phi according to the Klein Nishina equation

    Parameters
    ----------
    theta : float or array
    phi   : float or array
    ko    : float or array 
        The energy of the incident photon in eV 
    element : int (optional)
        Atomic number or the element scattering. If none is supplied, this function returns scattering from a free electron

    Returns
    -------
    d : float or array
        The differential scattering element for a given phi and theta
    """
    k = ko*mu / (mu + ko*(1 - np.cos(np.pi - theta)))
    c = 0.25*ro2*np.square(k/ko)*(k/ko + ko/k - 2.*np.square(np.sin(theta)*np.cos(phi)))
    if Z is not None:
        wavelength = Planck*speed_of_light/(ko*electron_volt)*1e10
        x = np.sin((np.pi - theta)/2.)/wavelength
        c = c*scatteringfunction[Z](x)
    return c

def thomson(theta, phi, ko, Z=None):
    """
    compute the thomson portion differential intensity of scattering for a given theta and phi according to the Klein Nishina equation

    Parameters
    ----------
    theta : float or array
    phi   : float or array
    ko    : float or array 
        The energy of the incident photon in eV 
    element : int (optional)
        Atomic number or the element scattering. If none is supplied, this function returns scattering from a free electron

    Returns
    -------
    d : float or array
        The differential scattering element for a given phi and theta
    """
    t = ro2*(1. - np.square(np.sin(theta)*np.cos(phi)))
    if Z is not None:
        wavelength = Planck*speed_of_light/(ko*electron_volt)*1e10
        x = np.sin((np.pi - theta)/2.)/wavelength
        t = t*np.square(formfactor[Z](x))
    return t

def differential_intensity(theta, phi, ko, Z=None):
    """
    compute the total differential intensity of scattering for a given theta and phi according to the Klein Nishina equation

    Parameters
    ----------
    theta : float or array
    phi   : float or array
    ko    : float or array 
        The energy of the incident photon in eV 
    element : int (optional)
        Atomic number or the element scattering. If none is supplied, this function returns scattering from a free electron

    Returns
    -------
    d : float or array
        The differential scattering element for a given phi and theta
    """
    c = compton(theta, phi, ko, Z)
    t = thomson(theta, phi, ko, Z)
    return c + t




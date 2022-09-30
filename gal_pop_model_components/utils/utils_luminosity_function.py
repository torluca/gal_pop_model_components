#! /usr/bin/env python

# Copyright (C) 2022 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import numpy
# External modules
import numpy as np
import scipy

# gal_pop_model_components imports


_LOGMAX = np.log(np.finfo(0.).max)


def compute_lower_truncation_scaled_schechter_random_variable(redshift_grid, apparent_magnitude_limit,
                                                              cosmology_object, m_star):
    """
    This function computes first the limit absolute magnitude at every redshift given the limit apparent magnitude,
    then subtract m_star and rescales it.

    :param redshift_grid: numpy.array: input redshift grid
    :param apparent_magnitude_limit: float: limiting apparent magnitude of sampled galaxies
    :param cosmology_object: astropy.cosmology: astropy.cosmology objects for computing distance modulus
    :param m_star: float or numpy.array: characteristic absolute magnitude, it can be constant or function of
                                         redshift_grid
    :return: lnxmin: numpy.array: absolute magnitude lower truncation of scaled Schechter random variable
    """
    lnxmin = apparent_magnitude_limit - cosmology_object.distmod(np.clip(redshift_grid, 1e-10, None)).value
    lnxmin -= m_star
    lnxmin *= -0.92103403719761827361

    return lnxmin


def gamma_function_integration_for_redshift(lnxmin, alpha):
    """
    This function performs the integration of the gamma function over the redshift grid for each limiting absolute
    magnitude lnxmin.

    :param lnxmin: numpy.array: array of limiting absolute magnitudes
    :param alpha: float or numpy.array: faint-end power-law slope
    :return:
    """

    def gamma_function_integrand(lnx, a):
        """
        Schechter's luminosity function can be expressed as gamma functions through appropriate change of variables:
        x^a e^-x

        :param lnx: numpy.array: random variable for changed variable x
        :param a: numpy.array: exponent for changed variable
        :return: numpy.array: gamma function integrand
        """

        return np.exp((a + 1) * lnx - np.exp(lnx)) if lnx < _LOGMAX else 0.

    gamma = np.empty_like(lnxmin)
    if numpy.isscalar(alpha):
        for i, _ in np.ndenumerate(gamma):
            gamma[i], _ = scipy.integrate.quad(gamma_function_integrand, lnxmin[i], np.inf, args=(alpha,))
    else:
        for i, _ in np.ndenumerate(gamma):
            gamma[i], _ = scipy.integrate.quad(gamma_function_integrand, lnxmin[i], np.inf, args=(alpha[i],))

    return gamma


def get_minimum_limiting_absolute_magnitude(galaxy_redshifts, apparent_magnitude_limit, cosmology_object, m_star):
    """
    This function computes the minimum limiting absolute magnitude for each galaxy redshift.

    :param galaxy_redshifts: numpy.array: sampled galaxy redshifts from luminosity function
    :param apparent_magnitude_limit: float: limiting apparent magnitude of sampled galaxies
    :param cosmology_object: astropy.cosmology: astropy.cosmology objects for computing distance modulus
    :param m_star: float or numpy.array: characteristic absolute magnitude, it can be constant or function of
                                         redshift
    :return x_min: numpy.array: minimum limiting absolute magnitude for each galaxy redshift
    """

    x_min = apparent_magnitude_limit - cosmology_object.distmod(galaxy_redshifts).value
    x_min -= m_star
    x_min *= -0.4
    if np.ndim(x_min) > 0:
        np.power(10., x_min, out=x_min)
    else:
        x_min = 10. ** x_min

    return x_min


def sample_from_schechter_function(alpha, x_min, x_max, size=None, scale=1., resolution=1000):
    """
    This function draws samples from the Schechter function expressed in a functional form of the kind x^alpha * e^x
    via a change of variable to use the properties of the gamma function.

    :param alpha: float or numpy.array: faint-end power-law slope
    :param x_min: numpy.array: minimum limiting absolute magnitude for each galaxy redshift
    :param x_max: numpy.array: maximum limiting absolute magnitude for each galaxy redshift
    :param size: int: output shape of samples. If size is None and scale is a scalar, a
                      single sample is returned. If size is None and scale is an array, an
                      array of samples is returned with the same shape as scale. Default is None.
    :param scale: float or numpy.array: scale factor for the returned samples. Default is 1.
    :param resolution: int: resolution of the inverse transform sampling spline.
    :return samples: numpy.array: samples drawn from the Schechter function.
    """

    if size is None:
        size = np.broadcast(x_min, x_max).shape or None

    lnx_min = np.log(x_min)
    lnx_max = np.log(x_max)

    lnx = np.linspace(np.min(lnx_min), np.max(lnx_max), resolution)

    pdf = np.exp(np.add(alpha, 1) * lnx - np.exp(lnx))
    cdf = pdf  # in place
    np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(lnx), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    t_lower = np.interp(lnx_min, lnx, cdf)
    t_upper = np.interp(lnx_max, lnx, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    lnx_sample = np.interp(u, cdf, lnx)

    samples = np.exp(lnx_sample) * scale

    return samples

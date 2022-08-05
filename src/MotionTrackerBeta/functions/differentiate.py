# Copyright 2022 Kristof Floch
 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pynumdiff
import pynumdiff.optimize


def differentiate(p, dt, parameters):
    """
    Calculates the velocity and the acceleration based on the given position data using the selected algortithm
    
    :param p: (np.array of floats, 1xN) time series to differentiate
    :param dt: (float) time step
    :param parameters: (dict) parameters for the differentiation 
    :return: ret : returns True if the differentiation was successful
    :return: ps : smoothed position
    :return: v : calculated velocity
    :return: a : calculated acceleration
    """

    if parameters[1] == "First Order Finite Difference":
        try:
            ps, v = pynumdiff.finite_difference.first_order(p, dt)
            v, a = pynumdiff.finite_difference.first_order(v, dt)
            return True, ps, v, a
        except:
            return False, 0, 0, 0

    elif parameters[1] == "Second Order Finite Difference":
        try:
            ps, v = pynumdiff.finite_difference.second_order(p, dt)
            v, a = pynumdiff.finite_difference.second_order(v, dt)
            return True, ps, v, a
        except:
            return False, 0, 0, 0

    elif parameters[1] == "Iterated First Order Finite Difference":
        try:
            ps, v = pynumdiff.finite_difference.first_order(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.finite_difference.first_order(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except:
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Median Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.mediandiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.mediandiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Mean Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.meandiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.meandiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Gaussian Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.gaussiandiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.gaussiandiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Butterworth Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.butterdiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.butterdiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Friedrichs Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.friedrichsdiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.friedrichsdiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Spline Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.splinediff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.splinediff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Iterative Total Variation Regularization with Regularized Velocity"
    ):
        try:
            parameters[3]["cg_maxiter"] = len(p)
            parameters[3]["scale"] = "small" if len(p) < 1000 else "large"
            ps, v = pynumdiff.total_variation_regularization.iterative_velocity(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.total_variation_regularization.iterative_velocity(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Velocity"
    ):
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Acceleration"
    ):
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Convex Total Variation Regularization with Regularized Jerk":
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Convex Total Variation Regularization with Sliding Jerk":
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk_sliding(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk_sliding(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Smoothed Acceleration"
    ):
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Spectral Derivative":
        try:
            ps, v = pynumdiff.linear_model._linear_model.spectraldiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.spectraldiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Savitzky-Golay Filter":
        try:
            ps, v = pynumdiff.linear_model._linear_model.savgoldiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.savgoldiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Polynomial Derivative":
        try:
            ps, v = pynumdiff.linear_model._linear_model.polydiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.polydiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Chebychev Polynomial Fit":
        try:
            ps, v = pynumdiff.linear_model._linear_model.chebydiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.chebydiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    else:
        return False, 0, 0, 0


def optimize_and_differentiate(p, dt, parameters):
    """Calculates the velocity and the acceleration based on the given position data using optimization to determine the ideal parameters of the delected differentiation algorithm"""
    gamma = np.exp(-1.6 * np.log(parameters[2]) - 0.71 * np.log(dt) - 5.1)

    if parameters[1] == "Iterated First Order Finite Difference":
        try:
            params, val = pynumdiff.optimize.finite_difference.first_order(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.finite_difference.first_order(p, dt, params)
            params, val = pynumdiff.optimize.finite_difference.first_order(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.finite_difference.first_order(
                v, dt, params, options={"iterate": True}
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Median Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.mediandiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.mediandiff(p, dt, params)
            params, val = pynumdiff.optimize.finite_difference.first_order(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.mediandiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Mean Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.meandiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.meandiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.meandiff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.meandiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Gaussian Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.gaussiandiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.gaussiandiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.gaussiandiff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.gaussiandiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Butterworth Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.butterdiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.butterdiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.butterdiff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.butterdiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Friedrichs Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.friedrichsdiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.friedrichsdiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.friedrichsdiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.friedrichsdiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Spline Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.splinediff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.splinediff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.splinediff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.splinediff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Iterative Total Variation Regularization with Regularized Velocity"
    ):
        try:
            options = {"cg_maxiter": len(p)}
            options["scale"] = "small" if len(p) < 1000 else "large"
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.iterative_velocity(
                p, dt, params=None, tvgamma=gamma, options=options
            )
            ps, v = pynumdiff.total_variation_regularization.iterative_velocity(
                p, dt, params, options
            )
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.iterative_velocity(
                v, dt, params=None, tvgamma=gamma, options=options
            )
            v, a = pynumdiff.total_variation_regularization.iterative_velocity(
                v, dt, params, options
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Velocity"
    ):
        try:

            params, val = pynumdiff.optimize.total_variation_regularization.velocity(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                p, dt, params
            )

            params, val = pynumdiff.optimize.total_variation_regularization.velocity(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Acceleration"
    ):
        try:
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.acceleration(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                p, dt, params
            )
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.acceleration(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] in {
        "Convex Total Variation Regularization with Regularized Jerk",
        "Convex Total Variation Regularization with Sliding Jerk",
    }:
        try:
            (params, val) = pynumdiff.optimize.total_variation_regularization.jerk(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                p, dt, params
            )
            (params, val) = pynumdiff.optimize.total_variation_regularization.jerk(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0

    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Smoothed Acceleration"
    ):
        try:
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.smooth_acceleration(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                p, dt, params
            )
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.smooth_acceleration(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Spectral Derivative":
        try:
            params, val = pynumdiff.optimize.linear_model.spectraldiff(
                p, dt, params=None, tvgamma=gamma
            )

            ps, v = pynumdiff.linear_model._linear_model.spectraldiff(p, dt, params)

            params, val = pynumdiff.optimize.linear_model.spectraldiff(
                v, dt, params=None, tvgamma=gamma
            )

            v, a = pynumdiff.linear_model._linear_model.spectraldiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Savitzky-Golay Filter":
        try:
            params, val = pynumdiff.optimize.linear_model.savgoldiff(
                p, dt, params=None, tvgamma=gamma
            )
            ps, v = pynumdiff.linear_model._linear_model.savgoldiff(p, dt, params)
            params, val = pynumdiff.optimize.linear_model.savgoldiff(
                v, dt, params=None, tvgamma=gamma
            )
            v, a = pynumdiff.linear_model._linear_model.savgoldiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Polynomial Derivative":
        try:
            params, val = pynumdiff.optimize.linear_model.polydiff(
                p, dt, params=None, tvgamma=gamma
            )
            ps, v = pynumdiff.linear_model._linear_model.polydiff(p, dt, params)
            params, val = pynumdiff.optimize.linear_model.polydiff(
                v, dt, params=None, tvgamma=gamma
            )
            v, a = pynumdiff.linear_model._linear_model.polydiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Chebychev Polynomial Fit":
        try:
            params, val = pynumdiff.optimize.linear_model.chebydiff(
                p, dt, params=None, tvgamma=gamma
            )
            ps, v = pynumdiff.linear_model._linear_model.chebydiff(p, dt, params)
            params, val = pynumdiff.optimize.linear_model.chebydiff(
                v, dt, params=None, tvgamma=gamma
            )
            v, a = pynumdiff.linear_model._linear_model.chebydiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    else:
        return False, 0, 0, 0
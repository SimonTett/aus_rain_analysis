"""
 distribution with linear covariates on location and scale. Code copied across from PIK repo for now.

This module provides a  distribution class similar  to  scipy.stats.rv_continuous
interface, but with support for linear covariates on both location and scale parameters.

Features:
    - Linear covariate dependence on location and scale
    - Compatible with scipy.stats API (pdf, logpdf, cdf, fit, etc.)

Code structure:
    - Covariate class with methods for fitting and distribution functions
    - Example usage demonstrating synthetic data generation and model fitting

Example usage: needs updating.
    >>> # Minimal example: fit a GEV with an intercept + 1 covariate on a small toy dataset

    >>> import numpy as np
    >>> from covariate import CovariateDistribution
    >>>
    >>> np.random.seed(42)
    >>> n = 200
    >>> X_loc = np.column_stack([np.ones(n), np.linspace(0, 1, n)])  # intercept + linear covariate
    >>> X_mu = np.ones((n, 1))  # constant log-scale (intercept only)
    >>> true_shape = 0.2
    >>> true_loc_coef = np.array([10.0, 1.5])
    >>> true_mu_coef = np.array([0.0])
    >>>
    >>> location = X_loc @ true_loc_coef
    >>> scale = np.exp(X_mu @ true_mu_coef)
    >>> from scipy.stats import genextreme
    >>> y = genextreme.rvs(c=true_shape, loc=location, scale=scale, size=n)
    >>>
    >>> # Fit the model (choose method via kwargs, e.g. method='Nelder-Mead') Note that NM is not a great choice.
    >>> gev = CovariateDistribution(X_loc=X_loc, X_mu=X_mu)
    >>> result = CovariateDistribution.fit(y, X_loc=X_loc, X_mu=X_mu, method='Nelder-Mead', options={'maxiter':1000})
    >>> # Inspect fitted parameters
    >>> print(result.shape, result.loc_coefficients, result.mu_coefficients)

Note: this example is intentionally small and self-contained. In practice prefer larger sample sizes
and set optimizer options/bounds appropriate to your problem. Also be aware that this module uses
`lmoments3` for an initial guess and `scipy.optimize.minimize` for fitting.
"""
# See https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.70075 for paper that deals with missing data when fitting GEV.
# Suspect approach is specific to GEV. SO, will need a GEV class that sits on top of covariate and handles missing data.

import numpy as np
import warnings
import typing
import scipy.special
import matplotlib.pyplot as plt
import logging
import scipy.optimize

from scipy.stats import genextreme, pearson3
import functools
import inspect
import textwrap

import lmoments3.distr
import xarray

my_logger = logging.getLogger(__name__)

class CovariateDistribution:
    """
    Distribution with linear covariates on location and mu (log scale).

    This class somewhat follows the scipy.stats.rv_continuous interface but extends it to support
    linear covariate effects on location and mu parameters.

    Attributes:
        shape: shape parameter (estimated during fitting)
        loc_coefficients: Coefficients for location linear predictor
        mu_coefficients: Coefficients for mu linear predictor
    """

    def __init__(self, c: typing.Optional[float] = None,
                 X_loc: np.ndarray = None,
                 X_mu: typing.Optional[np.ndarray] = None,
                 loc_coefficients: typing.Optional[np.ndarray] = None,
                 mu_coefficients: typing.Optional[np.ndarray] = None,
                 distribution:scipy.stats.rv_continuous = scipy.stats.norm):
        """
        Initialize covariate.

        Args:
            c  -- shape(s). Default None meaning no shape parameter.

            X_loc: Design matrix for location (n_samples, n_loc_features).
                   Typically, includes a column of ones for intercept.  If None  will be 1.
            X_mu: Design matrix for mu (log scale) (n_samples, n_mu_features).
                    If None will be 1.

            loc_coefficients: Coefficients for location linear predictor (length n_loc_features). If None, initialized to zeros.
            mu_coefficients: Coefficients for mu linear predictor (length n_mu_features). If None, initialized to loc_coefficients.
            distribution: scipy.stats distribution to use (default: scipy.stats.norm).

        """



        if isinstance(c,(list,np.ndarray)) and len(c) != 1:
            raise NotImplementedError("Currently only support single shape parameter. If your distribution has multiple shape parameters, modify this code to support.")
        elif isinstance(c,(list,np.ndarray)) and len(c) == 1:
            self.shape = float(c[0])
        elif isinstance(c,(np.number,float,int)):
            self.shape = float(c)
        else:
            self.shape = c



        # handle coefficients. If not provided are zero giving constant loc (=0) and scale (=1=np.exp(0)).
        if loc_coefficients is not None:
            self.loc_coefficients = np.asarray(loc_coefficients)
        else:
            self.loc_coefficients = np.zeros(self.X_loc.shape[1])

        if mu_coefficients is not None:
            self.mu_coefficients = np.asarray(mu_coefficients)
        else:
            self.mu_coefficients = np.zeros(self.X_mu.shape[1])

        # handle distribution
        self.distribution = distribution


        #self.n_samples = self.X_loc.shape[0]
        #self.n_loc_features = self.X_loc.shape[1]
        #self.n_mu_features = self.X_mu.shape[1]


    def design_matrices(self) -> tuple[np.ndarray,np.ndarray]:
        """
        Return default X_loc & X_mu design matrices compatible with loc/mu_coefficients.
        """
        X_loc= np.zeros((self.loc_coefficients.shape[0],1)) # loc =0
        X_mu = np.zeros((self.mu_coefficients.shape[0],1))  # scale = 1
        return X_loc, X_mu



    def _validate_parameters(
        self,
        loc_coef: np.ndarray,
        mu_coef: np.ndarray,
        shape: float
    ) -> typing.Tuple[bool, str]:
        """
        Validate parameter values are finite.

        Args:
            loc_coef: Location coefficients
            mu_coef: mu coefficients (log scale)
            shape: Shape parameter

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for NaN/Inf
        if np.any(~np.isfinite(loc_coef)) or np.any(~np.isfinite(mu_coef)):
            return False, "Non-finite coefficients"

        if not np.isfinite(shape) or np.isnan(shape):
            return False, "Non-finite shape parameter"

        return True, "A OK" # Everything looks good


    def fit_logpdf(
        self,
        y: np.ndarray,
        loc_coef: typing.Optional[np.ndarray]=None,
        mu_coef: typing.Optional[np.ndarray]=None,
        shape: typing.Optional[float] = None,
        verbose: bool = False,
        check_support: bool = False
    ) -> typing.Optional[float]:
        """
        Compute - sum log-probability density function when fitting data.

        Args:
            y: Observations (1D array, length n_samples)
            loc_coef: Location coefficients
            mu_coef: mu coefficients (log scale)
            shape: Shape parameter
            verbose: Print warnings about support violations
            check_support: Whether to apply support penalty for observations outside the distribution support

        Returns:
            log-probabilities (array)
        """
        raise NotImplementedError("fit_logpdf has been superseded. Code for support to be reused")
        y = np.asarray(y)
        if loc_coef is None:
            if self.loc_coefficients is None:
                raise ValueError("Location coefficients not provided and model not fit")
            loc_coef = self.loc_coefficients
        if mu_coef is None:
            if self.mu_coefficients is None:
                raise ValueError("mu coefficients not provided and model not fit")
            mu_coef = self.mu_coefficients
        if shape is None:
            if self.shape is None:
                raise ValueError("Shape parameter not provided and model not fit")
            shape = self.shape

        # Validate parameters
        is_valid, msg = self._validate_parameters(loc_coef, mu_coef, shape)
        if not is_valid:
            return None

        location = self._location(loc_coef)
        scale = self._scale(mu_coef)

        fdist = genextreme(c=shape,loc=location,scale=scale)
        if not check_support:
            logpdf = fdist.logpdf(y)
            sum_logpdf = -np.mean(logpdf)
            if not np.isfinite(sum_logpdf):  # outside support return a large number.
                sum_logpdf =None
            return float(sum_logpdf)
        # now deal with complex path to check support and apply penalty if outside support making the penalty function cts
        support_lb,support_ub = fdist.support()
        # check the lw < support_ub for all. Raise value error if not.
        if np.any(support_lb >= support_ub):
            raise ValueError("Invalid parameters: support lower bound is greater than or equal to upper bound for some observations. Check parameter values for numerical issues.")
        # NOTE: nextafter is used to avoid boundary issues; consider exposing the epsilon or using a more
        # principled numerical tolerance (e.g. relative to scale) rather than machine epsilon.
        support = (np.nextafter(support_lb,np.inf), np.nextafter(support_ub,-np.inf)) # reduce support by epsilon to avoid numerical issues
        yy = np.clip(y,support[0],support[1]) # clip data
        logpdf = fdist.logpdf(yy)
        if np.any(~np.isfinite(logpdf)):
            raise ValueError("Inf values in logpdf despite clipping")
        sum_logpdf = -np.mean(logpdf)
        L = yy != y # boolean array of which values were clipped
        if np.any(L): # have clipped yy values so need to apply penalty
            delta = np.abs(y-yy) # distance of original y values from clipped values
            # NOTE: The hard-coded gamma/tau choices below are heuristic. Consider making them parameters
            # (e.g. penalty_scale, penalty_tau) passed into `fit` or `fit_logpdf`, or computing them adaptively
            # from the data (e.g. median absolute deviation or mean scale). Also ensure they are numerically stable
            # for very large or small scales.
            gamma = 1000*(np.round(-logpdf.mean())+1.) # penalty fn prop to mean logpdf, but scaled up to ensure penalty is large enough to push optimization away from outside support
            tau = (scale).clip(1e-6,np.inf)# scaling on distance is avg of scale + std dev TODO: make this a parameter or compute based on data scale
            support_penalty = gamma * scipy.special.softplus(delta[L]/tau[L]) # smooth penalty for observations outside support
            support_penalty_sum = support_penalty.sum()/len(y) # average penalty per observation
            if not np.isfinite(support_penalty_sum):
                raise ValueError("Support penalty is not finite. Check parameter values and penalty function for numerical issues.")
            if support_penalty_sum < 0:
                raise ValueError("support_penalty_sum < 0")
            sum_logpdf +=  support_penalty_sum # add penalty to log-likelihood
            self.fit_outside_support_count += 1
            if verbose:
                print(f"Warning: {support_penalty_sum:.4e}  Added to log-likelihood.")





        return float(sum_logpdf)

    @staticmethod
    def mean_nll(
        params: np.ndarray,
        y: np.ndarray,
        sizes: typing.Tuple[int,int,int],
        X_loc: np.ndarray,
        X_mu: np.ndarray,
        weights: typing.Optional[np.ndarray] = None,
        distribution: typing.Callable  = scipy.stats.norm ,
        check_support: bool = True,
        penalty_out_support:typing.Optional[float] = None, #1e5,
        penalty_infinite_pdf:typing.Optional[float] = None

    ) -> float:
        """
        mean Negative log-likelihood (for optimization). (Mean to allow std optimization methods to work better)

        Args:
            param params: Concatenated array [loc_coef, mu_coef, shape]
            :param y: Observations
            :param sizes: sizes of respectively: location, mu & shape.  Used to extract from optimization
            :param X_loc location design matrix
            :param X_mu mu design matrix
            :param weights: weights array.
            :param distribution -  distribution to be used.  Default is Normal.
            :param check_support - if True  y values outside support have logpdf set to value of penalty_out_support
            :param penalty_out_support -- penalty value for values outside support when check_support is True. Default is 1e5
            :param penalty_infinite_pdf -- penalty value for infinite values returned from logpdf. Default is 1e4

            Both penalty_* values are checked against the largest finite value in the returned logpdf.
               If smaller then penalty values are increased with a warning message/


        Returns:
            Negative mean  log-likelihood (scalar)
        """

        if penalty_out_support is None:
            penalty_out_support = 1e5
        if penalty_infinite_pdf is None:
            penalty_infinite_pdf = 1e4


        # extract params from concatenated array
        loc_coef = params[:sizes[0]]
        mu_coef = params[sizes[0]:sizes[0]+sizes[1]]
        if sizes[2] > 0:
            shape = params[-sizes[2]:]
        else:
            shape = None
        # compute loc and scale from design matrices and coefficients
        loc = X_loc @ loc_coef
        scale= np.exp(X_mu @ mu_coef)
        y = np.asarray(y) # make sure we have a numpy array for the data to keep type checker happy.
        # apply guess scale and loc to transform data


        if shape is None:
            fdist = distribution(loc=loc,scale=scale)
        else:
            fdist = distribution(c=shape, loc=loc, scale=scale)

        # compute support
        support_lb, support_ub = fdist.support()
        # check the lw < support_ub for all. Raise value error if not.
        if np.any(support_lb >= support_ub):
            raise ValueError(
                "Invalid parameters: support lower bound is greater than or equal to upper bound for some observations. Check parameter values for numerical issues.")
        # then generate a very large penalty if outside support
        indx_outside = (y > support_ub) | (y < support_lb) # values above upper bound or lower than lower bound.
        count_outside = indx_outside.sum()
        if count_outside > 0:
            my_logger.debug(f"{count_outside} values outside support bounds")
            if not check_support:
                my_logger.warning("Outside support -- returning penalty")
                return float(np.finfo(np.array([1.0]).dtype).max)  # return a very large number if outside support.


        logpdf = fdist.logpdf(y) # compute log pdf for all  observations even when not in  support.
        indx = np.isfinite(logpdf) # good values -- bad values outside support or guess is a long way from loc.
        if np.any(indx): # got some finite values so can test penalty values
            mx_value = -logpdf[indx].min() # smallest value; take -ve as penalty gets applied to logpdf with - value
            if( mx_value > penalty_out_support) and check_support: # big value and are checking support
                my_logger.warning(f" -min(logpdf) = {mx_value:.3g} > {penalty_out_support:.3g}. Setting penalty_out_support to {mx_value*10:.3g}.")
                penalty_out_support = mx_value*10
            if mx_value > penalty_infinite_pdf:
                my_logger.warning(f"-min(logpdf) = {mx_value:.3g}  > {penalty_infinite_pdf:.3g}. Setting penalty_infinite_pdf to {mx_value*10:.3g}.")
                penalty_infinite_pdf = mx_value*10
        # end checking that penalty values are OK
        # add on penalty vaues
        logpdf[~indx] = -penalty_infinite_pdf
        logpdf[indx_outside] = -penalty_out_support # set values outside the support to a large penalty value

        if weights is None:
            mn_logpdf = logpdf.mean()
        else:
            mn_logpdf = np.average(logpdf,weights=weights) # weighted average.

        mn_logpdf = -float(mn_logpdf)




        my_logger.debug(f"mn_logpdf: {mn_logpdf:.3f} (shape={shape}, loc_coef={loc_coef}, mu_coef={mu_coef})")
        return float(mn_logpdf)



    @classmethod
    def fit(cls,
        distribution: scipy.stats.rv_continuous,
        y: np.ndarray,
        X_loc: typing.Optional[np.ndarray] = None,
        X_mu: typing.Optional[np.ndarray] = None,
        guess: typing.Optional[dict] = None,
        weights:typing.Optional[np.ndarray] = None,
        check_support: bool = True,
        return_params: bool = False,
        raise_error: bool = True,
        penalty_out_support: typing.Optional[float] = None,
        penalty_infinite_pdf: typing.Optional[float] = None,
        **kwargs
    ) -> tuple["CovariateDistribution",dict]|tuple[np.ndarray,np.ndarray,np.ndarray,float,float,float]:
        """
        Fit GEV distribution with covariates to data returns shape, location and scale parameters & dict of fit information.

        Args:
            distribution: scipy.stats distribution to use (default: norm)
            y: Observations (1D array)
            Following arguments are optional
            X_loc -- If None will be set 1 * no of data pts
            X_mu -- if None will be set to X_loc
            guess -- contains guess for optimal values.
              Should contain:
                c:  Initial guess for shape parameter. Only provide if your dist takes a shape parameter.
                loc_coefficients : Initial guess for location parameter. Only provide if your dist takes a location parameter.
                mu_coefficients: Initial guess for location parameter. Only provide if your dist takes a location parameter.
             If guess is  None, will use lmoments3 fit to get an initial guess for the  parameters (if supported for this distribution).

            weights: weights for each observation. Currently, not implemented.
            check_support: Whether to apply  penalty for observations outside the distribution support
            return_params: If True return only parameters (location, mu, shape) as numpy arrays,
              and AIC, nll & KS stat as floats. Here to support apply_ufunc.

            raise_error -- if True raise errors otherwise warn.
            penalty_infinite_pdf -- penalty for infinite values. Passed through to mean_nll.
            penalty_out_support -- penalty for values outside support. Passed through to mean_nll.

            **kwargs: Additional options passed through to minimizer function.
            See scipy.optimize.minimize but includes
                - method: Optimization method You will probably need to provide this but it is very data/distribution dependant.
                - bounds: Custom bounds for optimization
            DO NOT PROVIDE args or x0 in kwargs, as these are set by the function itself.

        Returns: CovariateDistribution & fit_info as dict with following keys:
            n_samples: number of samples used to fit the data
            n_params: total number of parameters fitted (length of loc_coef + length of mu_coef + len(shape) if shape is not None)
            neg_log_likelihood: negative log likelihood for best fit
            AIC: Akaike Information Criterion for the fitted model (2*n_params + 2*neg_log_likelihood)
            ks: Result of KS test.
            success: True if optimizer converged successfully, False otherwise
            message: Message from optimization
            resultObj: Full result object from scipy.optimize.minimize -- details


        """
        y = np.asarray(y)
        n_samples = y.shape[0]

        if X_loc is None:
            X_loc = np.ones((n_samples,1))

        if X_mu is None:
            X_mu = X_loc

        features_loc = X_loc.shape[1]
        features_mu = X_mu.shape[1]

        # remove missing data
        indx = ~np.isnan(y)
        y=y[indx]
        X_loc = X_loc[indx,:]
        X_mu = X_mu[indx,:]
        if weights is not None:
            weights = weights[indx]




        # check ranks = no of features to avoid perfect collinearity issues in optimization. Move to fit as matters here..
        if np.linalg.matrix_rank(X_loc) < features_loc:
            raise ValueError(f"X_loc has  collinearity (rank < {features_loc}). Remove redundant features or reformulate.")
        if np.linalg.matrix_rank(X_mu) < features_mu:
            raise ValueError(f"X_mu has  collinearity (rank < {features_mu}). Remove redundant features or reformulate.")




        scipy_to_lmoments_mapping = dict(
            expon = 'exp',
            gamma = 'gam',
            genextreme = 'gev',
            genpareto = 'gpa',
            gumbel_r = 'gum',
            norm = 'nor',
            pearson3 = 'pe3',
            weibull_min = 'wei'
        ) # mapping of scipy names to lmoments3 names for initial guess.
        # Note that some distributions may not have a direct equivalent in lmoments3, for now will raise an error.
        # If have actual use case will implement alt way of getting frist guess.
        # first guess for optimization is important for convergence, so we use lmoments3 fit to get an initial guess for shape, loc and scale parameters.
        # The shape parameter is also initialized from the lmoments3 fit. Hopefully this makes convergence more robust.
        if guess is None or (len(guess)==0): # any of the initial things are None so need to get first guess from lmoments3 fit
            lmoments3_name = scipy_to_lmoments_mapping.get(distribution.name,None)
            if lmoments3_name is None:
                raise ValueError(f"Distribution {distribution.name} not supported for initial guess. Supported distributions are: {list(scipy_to_lmoments_mapping.keys())}. Consider implementing an alternative method for getting an initial guess for this distribution.")
            first_guess_params = getattr(lmoments3.distr,lmoments3_name).lmom_fit(y) # use lmoments fit for initial guess of shape, loc and mu
            guess={}
            initial_loc_coef = np.zeros(features_loc)
            initial_loc_coef[0] = first_guess_params['loc']
            guess['loc_coefficients'] =initial_loc_coef # Start with  fit location as intercept guess

            initial_mu_coef = np.zeros(features_mu)
            initial_mu_coef[0] = np.log(first_guess_params['scale'] ) # Start with log of std dev as intercept guess
            guess["mu_coefficients"] = initial_mu_coef


            if 'c' in first_guess_params:
                guess["c"] = np.atleast_1d(first_guess_params['c'])
                # Start with movements3 shape as guess and ensure shape is 1D array for concatenation below


        guess_names = ['loc_coefficients', 'mu_coefficients']
        if distribution.shapes is not None:
            guess_names += ['c'] # TODO . Actually use the names here.
        # directly looking for k as want to trigger error if not present. Likely because of user error.
        initial_guess = [np.atleast_1d(guess[k]) for k in guess_names ]

        initial_guess = np.concatenate(initial_guess)
        lens = [len(np.atleast_1d(guess[k])) for k in guess_names]
        if distribution.shapes is None:
            lens = np.concatenate([lens,[0]])
        # empty lists when don't  have a shape.
        n_params = len(initial_guess)
        my_logger.debug(f"Starting fit with {n_samples} samples, {n_params} parameters")


        # Minimize negative log-likelihood
        args = ( y, lens, X_loc, X_mu)
        fn = functools.partial(CovariateDistribution.mean_nll,
                 distribution=distribution,  check_support=check_support,
                               penalty_out_support=penalty_out_support,
                               penalty_infinite_pdf=penalty_infinite_pdf) # roll kwrd args into function
        resultObj = scipy.optimize.minimize(fn, initial_guess, args=args,**kwargs)
        # Extract params
        # now work out the sum (not mean) of the log-likelihood at the fitted parameters for reporting
        neg_log_likelihood = resultObj.fun * len(y)
        AIC = 2 * n_params + 2 * neg_log_likelihood
        my_logger.debug(f" optimization result: mean nll={resultObj.fun:.2f}")

        # Extract fitted parameters
        x = resultObj.x
        x = np.asarray(x) # make sure we have a numpy array to keep type checker happy.
        loc_coefficients = x[:features_loc]
        mu_coefficients = x[features_loc:features_mu+features_loc]
        if len(x) == features_loc + features_mu: # No shape parameter
            dist_shape = None
        else:
            dist_shape = x[-len(x)+features_loc+features_mu:]
        my_logger.debug(f"Fit complete:  nll={resultObj.fun:.4f}")
        cov_dist = CovariateDistribution(dist_shape, X_loc, X_mu, loc_coefficients, mu_coefficients,
                                         distribution=distribution)
        if weights is not None:
            ks_result = None  # Need to implement weighted KS test if want to report this.  For now just return None.
        else:
            u = cov_dist.compute_pit(y, X_loc, X_mu)
            order = np.argsort(u)
            u_sorted = u[order]  # make it a uniform distribution
            ks_result = float(scipy.stats.kstest(u_sorted, scipy.stats.uniform.cdf).pvalue)
        # did we successfully optimize?
        if not resultObj.success:
            message = "Minimization failed: " + resultObj.message + ". Try providing a better initial guess or adjusting optimization options."
            if raise_error:
                raise ValueError(message)
            else:# failed to optimise so set everything to Nan
                neg_log_likelihood = np.nan
                ks_result = np.nan
                AIC=np.nan
                loc_coefficients[:]=np.nan
                mu_coefficients[:]=np.nan
                dist_shape[:]=np.nan
                cov_dist = CovariateDistribution(dist_shape, X_loc, X_mu, loc_coefficients, mu_coefficients,
                                                 distribution=distribution)
                my_logger.warning(message)


        if return_params:
            return loc_coefficients, mu_coefficients, dist_shape, AIC,neg_log_likelihood,ks_result

        fit_info = dict(
            n_samples = n_samples,
            n_params = n_params,
            neg_log_likelihood = neg_log_likelihood,
            AIC = AIC,
            KS = ks_result,
            success = resultObj.success,
            message = resultObj.message,
            resultObj = resultObj,
        )



        return cov_dist,fit_info



    type_coefficients = typing.Literal['loc','mu']
    def _predict_dist_param(self,
                            attribute_name:type_coefficients,
                            X: np.ndarray) -> np.ndarray:
        """
            Predict distribution parameter  from design matrix.
        :param attribute_name:  Name of attribute to use if X not provided
        :param X: design matrix to use for prediction. If None, uses training design matrix (from attribute_name).
        :return: Predicted value of distribution parameter
        """
        if self.loc_coefficients is None:
            raise ValueError("Model must be fit before prediction")


        coeffts = getattr(self, 'loc_coefficients' if attribute_name == 'loc' else 'mu_coefficients')
        return X @ coeffts

    def predict_location(self, X_loc:np.ndarray) -> np.ndarray:
        """
        Predict location parameter for new design matrix.

        Args:
            X_loc: Design matrix (n_new, n_loc_features).


        Returns:
            Location parameter array
        """
        loc = self._predict_dist_param('loc', X_loc)

        return loc

    def predict_scale(self, X_mu:np.ndarray) -> np.ndarray:
        """
        Predict scale parameter for new design matrix.

        Args:
            X_mu: Design matrix (n_new, n_scale_features) for mu (log scale) param.


        Returns:
            Scale parameter array
        """
        mu = self._predict_dist_param( "mu",X_mu)
        return np.exp(mu)

    def compute_pit(self, y: np.ndarray, X_loc:np.ndarray, X_mu:np.ndarray) -> np.ndarray:
        """
        Compute Probability Integral transform (PIT) values u_i = F(y_i | params_i) using CovariateDistribution.cdf.
        :param y -- data values


        Returns:
            u: 1-D array of PIT values in [0,1] PIT transforms from known distribution to uniform
        """
        y = np.asarray(y).ravel()

        # Call cdf
        u = self.cdf(y,X_loc,X_mu)

        return u

    def qq_pit_plot(self,
                    y: np.ndarray,
                    X_loc:np.ndarray,
                    X_mu:np.ndarray,
                    bins: int = 20,

                    ax: typing.Optional[typing.Any] = None,
                    **fig_kwrds) -> dict[str, typing.Any]:
        """
        Produce PIT histogram and PIT-based Q-Q plot with optional parametric bootstrap envelope.
        Sadly statsmodels does not work with non-constant loc/scale values.
        :param y: data


        Returns a dict with keys:
          - p: uniform plotting positions
          - u_sorted: sorted observed PIT values
          - lower, upper: envelope arrays (None if n_sims <= 0)
          - fig, axe_hist, ax_qq: plotting axes (if created)
        """
        y = np.asarray(y).ravel()
        n = y.size



        # observed PIT
        u = self.compute_pit(y,X_loc,X_mu)
        order = np.argsort(u)
        u_sorted = u[order]
        p = (np.arange(1, n + 1) - 0.5) / n

        # prepare plotting
        created_fig = False
        subplot_kwargs = dict(nrows=1, ncols=2, figsize=(10, 4),clear=True, layout="constrained")
        subplot_kwargs.update(fig_kwrds)
        if ax is None:

            fig, axes = plt.subplots(**subplot_kwargs)
            ax_hist, ax_qq = axes
            created_fig = True
        else:
            ax_hist = ax
            ax_qq = ax.twinx() if hasattr(ax, "twinx") else ax

        # work out ks-stat to display
        ks_result = scipy.stats.kstest(u_sorted, scipy.stats.uniform.cdf)

        # PIT histogram
        ax_hist.hist(u, bins=bins, density=True, alpha=0.65, color="C0", edgecolor="k")
        ax_hist.set_xlabel("PIT (u)")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"PIT histogram (should be ~Uniform KS test = {ks_result.pvalue:.3f})")



        # QQ plot (sorted PIT vs uniform plotting positions)
        ax_qq.plot(p, u_sorted, marker=".", linestyle="none", label="Observed")
        ax_qq.plot(p, p, color="k", linestyle="--", label="Ideal")
        ax_qq.set_xlabel("Theoretical quantiles (Uniform)")
        ax_qq.set_ylabel("Empirical sorted PIT")
        ax_qq.set_title("PIT Q-Q plot")
        ax_qq.legend()

        if created_fig:
            fig.show()



        return {
            "p": p,
            "u_sorted": u_sorted,
            "ks_result": ks_result,
            "fig": (fig if created_fig else None),
            "ax_hist": ax_hist,
            "ax_qq": ax_qq,
        }

    def _gen_fn(self,fn_name,
                x: np.ndarray,
                X_loc: np.ndarray,
                X_mu: np.ndarray) -> np.ndarray:
        """
        Generalized function to compute distribution functions (CDF, SF, etc.)

        Args:
            fn_name: Name of Function from scipy.stats.whatever (e.g. cdf, sf)
            x: Values to evaluate function at (e.g. quantiles for pdf/logpdf, observations for cdf/sf)
            X_loc: Design matrix for location
            X_mu: Design matrix for mu (log scale)

        Returns:
            Function values (same shape as y)
        """
        if self.loc_coefficients is None:
            raise ValueError("Model must be fit before prediction")


        location = self.predict_location(X_loc)
        scale = self.predict_scale(X_mu)
        shape = self.shape
        with warnings.catch_warnings(action='ignore'):
            fn = getattr(self.distribution,fn_name) # get the function wanted.
            if isinstance(x,(np.ndarray,list,tuple)) and False: # turning this off for now. Need to sort out wrapped stuff.
                x = np.asarray(x).reshape(-1, 1)


            if shape is None: # no shape for this distribution
                return fn(x,  loc=location, scale=scale)
            else:
                return fn(x, shape, loc=location, scale=scale)

    def _gen_fn_no_x(self,fn_name,
                X_loc: np.ndarray,
                X_mu: np.ndarray) -> typing.Any:
        """
        Generalized function to compute things like support, median etc

        Args:
            fn_name: Function from scipy.stats.whatever (e.g. support)
            shape: Shape parameter (if None, uses fitted shape)
            X_loc: Design matrix for location (default: training data)
            X_mu: Design matrix for mu (log scale) (default: training data)

        Returns:
            Function values
        """
        if self.loc_coefficients is None:
            raise ValueError("Model must be fit before prediction")


        location = self.predict_location(X_loc)
        scale = self.predict_scale(X_mu)
        shape = self.shape
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            fn = getattr(self.distribution,fn_name)
            if shape is None:
                return fn(loc=location, scale=scale) # no shape for this distribution
            else:
                return fn(shape, loc=location, scale=scale) # this dist has a shape.



# Code to add wrapper methods for continuous_rv functions (pdf, cdf, etc.) to the CovariateDistribution class. These methods will delegate to the existing _gen_fn which already handles validation and prediction of parameters.
# This avoids code duplication and ensures consistent behaviour across all distribution functions.
# Code AI generated.



_methods = ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf", "interval", "moment"]
# methods that take an x input.
_methods_no_x = ["support", "median", 'stats', "entropy", "mean", "var", "std"]
# methods that don't take an x input and thus need a different wrapper signature and docstring

def _make_doc(underlying_doc: typing.Optional[str], fn_name: str) -> str:
    base = textwrap.dedent(underlying_doc or "")
    extra = (
        "\n\nWrapper notes (CovariateDistribution):\n"
        "- This method wraps `scipy.stats` and adds covariate support.\n"
        "- Extra parameters: `shape` (float, optional), `X_loc` (design matrix for location), "
        "`X_mu` (design matrix for mu / log-scale).\n"
        "- If any of `shape`, `X_loc`, `X_mu` is `None`, the fitted value from the instance is used.\n"
        "- `location` and `scale` are computed from the fitted coefficients and provided design matrices.\n"
        "\nReturn value: NumPy array (same shape as input) or scipy return as documented.\n"
    ).format(fn_name)
    return base + extra

# python
def _make_wrapper(underlying: str, takes_x: bool) -> typing.Callable:
    """
    Factory function to generate wrapper methods for scipy.stats distribution functions with covariate support.
    
    Creates a wrapper that adds covariate-dependent location and scale parameters to scipy.stats
    distribution methods (pdf, cdf, etc.). The wrapper delegates to either _gen_fn (for methods
    that take an evaluation point x) or _gen_fn_no_x (for methods like support, median).
    
    Handles parameter construction and signature management to make wrapped methods appear
    as native CovariateDistribution methods.
    
    Args:
        underlying (str): Name of the scipy.stats.rv_continuous method to wrap (e.g., 'pdf', 'cdf', 'support').
        takes_x (bool): If True, wrapper accepts x parameter (evaluation point). If False, wrapper
                       does not accept x (used for methods like support, median, stats).
    
    Returns:
        callable: Wrapper function with:
            - self: CovariateDistribution instance
            - x (optional if takes_x=True): Evaluation point(s) 
            - shape (keyword-only, optional): Shape parameter (uses fitted value if None)
            - X_loc (keyword-only, optional): Design matrix for location (uses training data if None)
            - X_mu (keyword-only, optional): Design matrix for mu/log-scale (uses training data if None)
            - **kwargs: Passed to underlying scipy.stats method
            
            Returns: Result from underlying scipy.stats method, with location and scale computed
                    from covariate design matrices and fitted coefficients.
    
    Notes:
        - The wrapper's __doc__ and __signature__ are set to appear native to CovariateDistribution
        - Parameter order is handled via inspect.Signature for clean introspection
        - Location and scale are computed dynamically via _gen_fn/_gen_fn_no_x for each call
    """
    # Build inspect.Parameter objects to define the wrapper's signature programmatically.
    # This allows us to create clean function signatures that appear native to CovariateDistribution.
    self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    # x is optional and can be passed positional or keyword; used for evaluation points (pdf, cdf, etc.)
    x_param = inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
    # shape, X_loc, X_mu are keyword-only to avoid confusion with positional args; all optional
    shape_param = inspect.Parameter("shape", inspect.Parameter.KEYWORD_ONLY, default=None)
    X_loc_param = inspect.Parameter("X_loc", inspect.Parameter.KEYWORD_ONLY, default=None)
    X_mu_param = inspect.Parameter("X_mu", inspect.Parameter.KEYWORD_ONLY, default=None)
    # **kwargs captures any additional arguments for the underlying scipy.stats method
    kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)

    # Build parameter list: methods that take_x include the x_param; others omit it.
    # This is stored separately to assign to _wrapper.__signature__ later for introspection.
    if takes_x:
        params = [self_param, x_param, shape_param, X_loc_param, X_mu_param, kwargs_param]
    else:
        params = [self_param, shape_param, X_loc_param, X_mu_param, kwargs_param]

    # Define the actual wrapper function. Split into two cases to match expected signatures.
    if takes_x:
        # For methods like pdf, cdf, sf that need an evaluation point x (e.g., observations or quantiles)
        @functools.wraps(underlying)
        def _wrapper(self, x: typing.Any,
                     X_loc: np.ndarray,
                     X_mu: np.ndarray,
                     _underlying=underlying,  # capture method name via default; avoids closure issues with loop
                     **kwargs):
            # Delegate to _gen_fn which computes location/scale from covariates and calls the scipy method
            return self._gen_fn(_underlying, x,  X_loc, X_mu, **kwargs)
    else:
        # For methods like support, median, stats that don't need x (computed from distribution params only)
        @functools.wraps(underlying)
        def _wrapper(self,
                     X_loc: np.ndarray,
                     X_mu: np.ndarray,
                     _underlying=underlying,  # capture method name via default; avoids closure issues with loop
                     **kwargs):
            # Delegate to _gen_fn_no_x which handles methods that don't take x
            return self._gen_fn_no_x(_underlying,  X_loc, X_mu, **kwargs)

    # Set the wrapper's docstring by combining the underlying scipy.stats docstring with our custom wrapper notes
    _wrapper.__doc__ = _make_doc(getattr(underlying, "__doc__", None), getattr(underlying, "__name__", "rv_continuous"))
    # Set the wrapper's signature explicitly so help() and introspection tools show the right parameters
    _wrapper.__signature__ = inspect.Signature(params)
    return _wrapper


def _attach_wrappers(target_cls):
    """
    Attach wrappers to scipt
    :param target_cls:
    :return:
    """
    for name in _methods:
        try:
            underlying = getattr(scipy.stats.rv_continuous, name)
            setattr(target_cls, name, _make_wrapper(name, takes_x=True))
        except AttributeError:
            my_logger.warning(f"{name} not found in scipy.stats.rv_continuous so not wrapped")

    for name in _methods_no_x:
        try:
            underlying = getattr(scipy.stats.rv_continuous, name)
            setattr(target_cls, name, _make_wrapper(name, takes_x=False))
        except AttributeError:
            my_logger.warning(f"{name} not found in scipy.stats.rv_continuous so not wrapped")





# Attach wrappers to the class
_attach_wrappers(CovariateDistribution)

def dist_fit_wrapper(y:np.ndarray, X_loc:np.ndarray,X_mu:np.ndarray,
                     weights:typing.Optional[np.ndarray],
                     guess_c,guess_loc,guess_mu,
                     distribution:scipy.stats.rv_continuous = scipy.stats.norm,
                     **kwargs):

    guess=dict(c=guess_c, loc_coefficients=guess_loc, mu_coefficients=guess_mu)
    if distribution.shapes is None:
        guess.pop('c') # no shape parameter for this distribution so remove it from the guess dict.
    # remove any that are None.  This is needed because the covariate distribution fit function doesn't like None values in the guess dict.
    guess = {k:np.asarray(v) for k,v in guess.items() if v is not None}
    if guess: # got some start guess.
        kwargs.update(guess=guess)
    fit = CovariateDistribution.fit(distribution, y, X_loc=X_loc, X_mu=X_mu,
                                              weights=weights,
                                              return_params = True,**kwargs)

    return fit

def xarray_dist_fit(
        data_array: xarray.DataArray,
        distribution: scipy.stats.rv_continuous = scipy.stats.norm,
        dim:  str = 'time',
        X_loc: typing.Optional[xarray.DataArray] = None,
        X_mu: typing.Optional[xarray.DataArray] = None,
        guess: typing.Optional[typing.Union[xarray.Dataset,dict[str,float|np.ndarray]]] = None,
        weights: typing.Optional[xarray.DataArray] = None,
        use_dask: bool = False,
        raise_error:bool = True,
        penalty_infinite_pdf: typing.Optional[float] = None,
        penalty_outside_support: typing.Optional[float] = None,
        **kwargs
) -> xarray.Dataset:
    #
    """
    Fit a distribution  to xarray data

    :param data_array: dataArray for which GEV is to be fit
    :param dist: distribution to fit Should be a scipy.stats.rv_continuous dist. Default is normal distribution.
    :param X_loc: design matrix for the location parameters.  Last dimension will be used for the fit coefficient dimensions.
      If None will be set a 1D unit matrix.
    :param X_mu: design matrix for the mu parameters. mu = log(scale). Should be compatible with X_loc.
      If None will be set to X_loc
    :param guess  Either dataSet of initial guess or dict of np.ndarrays.  If None left alone.
          Should contain:
            c:  Initial guess for shape parameter. Only provide if your dist takes a shape parameter.
            loc_coefficients : Initial guess for location parameter. Only provide if your dist takes a location parameter.
            mu_coefficients: Initial guess for location parameter. Only provide if your dist takes a location parameter.
    :param weights: Weights for each sample. If not specified, no weighting will be done.
    :param dim: The dimension(s) over which to collapse. Default is time.
    :param use_dask: If True use dask to do the fitting. Will give parallel running if you have multi processing active.
    :param kwargs: any kwargs passed through to the fitting function
    :return: a dataset containing:
        Parameters -- the parameters of the fit; location_coefficients, mu_coefficients, shape
        #StdErr -- the standard error of the fit -- same parameters as Parameters
        #Cov -- the covariance matrix of the fit -- same parameters as Parameters
        nll -- negative log likelihood of the fit -- measure of the quality of the fit
        AIC -- Akaike information criteria.
        #KS -- KS test result
    """



    # Deal with design matrix.
    n_sample = len(data_array.coords[dim])
    if X_loc is None:
        X_loc = xarray.DataArray(np.ones((n_sample, 1)), dims=[dim, 'coeff'], coords={dim: data_array.coords[dim]})
        my_logger.debug("Setting X_loc to 1")
    if X_mu is None:
        X_mu = X_loc
        my_logger.debug(f"Setting X_mu to X_loc")

    output_dim = set(X_loc.dims) - set(data_array.dims) # see what the output_dim is for location
    my_logger.debug(f"Setting output_dim to {output_dim}")

    # check (AI code) that X_mu and X_loc are compatible
    OK = (X_loc.dims == X_mu.dims) and (X_loc.sizes == X_mu.sizes)
    if not OK:
        raise ValueError('X_loc and X_mu must have the same dimensions and sizes')

    output_dim = 'coeff'
    input_core_dims = [ # set up input core-dims
            [dim], # data
            [dim,output_dim], # X_loc
            [dim,output_dim],# X_mu
           ]

    if weights is None:
        input_core_dims.append([])
    else:
        input_core_dims.append([dim]) # weights

    # handle guess. Which implies adding guess_args and the core_dims
    if guess is None:
        guess={} # empty dict
    guess_args = []
    for k,dim_name in zip( ["c", "loc_coefficients", "mu_coefficients"],['shape_coeff_in', 'loc_coeff_in', 'mu_coeff_in']):
        guess_data_array = guess.get(k, None)

        if guess_data_array is None or guess_data_array.ndim==0:
            input_core_dims.append([])
        else:
            # hack for dims
            rename_dict = dict(shape_coeff='shape_coeff_in',loc_coeff='loc_coeff_in',mu_coeff='mu_coeff_in',coeff=dim_name)
            rename_dict = {k:rename_dict[k] for k in guess_data_array.dims if k in rename_dict}
            guess_data_array = guess_data_array.rename(rename_dict)
            input_core_dims.append(list(set(guess_data_array.dims)-set(data_array.dims)))

        guess_args.append(guess_data_array)

    # broadcast them all to the same size



    output_core_dims = [["loc_"+output_dim],# location coefficients
                        ["mu_"+output_dim], # parameter coefficients
                        ['shape_'+output_dim], # shape coefficients
                        [], # nll
                        [], # AIC
                        [] # KS result
                        ]

    # set up kwargs passed to function.
    kwargs.update(distribution=distribution)
    kwargs.update(raise_error=raise_error)
    kwargs.update(penalty_infinite_pdf=penalty_infinite_pdf)
    kwargs.update(penalty_out_support=penalty_outside_support)


    if use_dask: # setup dask args
        output_sizes = dict(loc_coeff=X_loc.sizes[output_dim],
                            mu_coeff=X_mu.sizes[output_dim],
                            shape_coeff=1, ) # expected output sizes needed by dask
        output_dtypes = [float]*6
        extra_args = dict(dask='parallelized',
                          dask_gufunc_kwargs=dict(allow_rechunk=True,output_sizes=output_sizes),
                          output_dtypes=output_dtypes)
        my_logger.debug('Using dask')
    else:
        extra_args = {}



    my_logger.debug('Doing fit')


    loc,mu,shape, nll, AIC, ks = xarray.apply_ufunc(dist_fit_wrapper, data_array,X_loc,X_mu, weights,*guess_args,
                                                                               input_core_dims=input_core_dims,
                                                                               output_core_dims=output_core_dims,
                                                                               vectorize=True,
                                                                               kwargs=kwargs,
                                                                               **extra_args
                                                                               )
    loc = loc.assign_coords(loc_coeff=X_loc.coords['coeff'].values)
    mu = mu.assign_coords(mu_coeff=X_mu.coords['coeff'].values)
    shape = shape.assign_coords(shape_coeff=[0])
    my_logger.debug('Done fit. Making dataset')

    data_array = xarray.Dataset(dict(location=loc,mu=mu,shape=shape,nll=nll,AIC=AIC))
    if use_dask:
        my_logger.debug('Computing for dask')
        logger = logging.getLogger('distributed.utils_perf')
        logger.setLevel('ERROR')
        data_array = data_array.compute()
        my_logger.debug('Done dask GEV computation')

    return data_array


if __name__ == "__main__":
    # Example usage (updated): small, quick demonstration for interactive testing
    my_logger.setLevel('DEBUG')
    import numpy as np
    print(np.__version__)
    np.random.seed(42)


    # Generate synthetic data
    n_samples = 2000
    X_loc = np.column_stack([np.ones(n_samples), np.linspace(0, 1, n_samples)])  # Intercept + linear covariate
    #X_mu = np.ones((n_samples, 1))
    X_mu = X_loc # mu has same design matrix as location.

    true_shape = 0.1
    true_loc_coef = np.array([10.0, .5])
    true_mu_coef = np.array([0.0,0.1]) # want scales to be around 1

    location = X_loc @ true_loc_coef
    scale = np.exp(X_mu @ true_mu_coef)

    # Sample from GEV
    gev_y = genextreme.rvs(c=true_shape, loc=location, scale=scale, random_state=42, size=n_samples)

    # Fit the model: show how to pass optimization kwargs
    print("Fitting GEV with covariates (small demo)...")
    gdist = scipy.stats.genextreme
    gev_dist,optResult_gev = CovariateDistribution.fit(gdist, gev_y, X_loc=X_loc, X_mu=X_mu)#,method='CG')
    gev_dist_params = CovariateDistribution.fit(gdist, gev_y, X_loc=X_loc, X_mu=X_mu, return_params=True,
                                                guess=dict(c=0.0,loc_coefficients=[gev_y.mean(),0.0],mu_coefficients=[np.log(gev_y.std()),0.0]))

    print("\n=== Results ===")
    print(f"True shape: {true_shape:.6f}")
    print(f"Fitted shape: {float(gev_dist.shape):.6f}")
    print(f"\nTrue location coefficients: {true_loc_coef}")
    print(f"Fitted location coefficients: {gev_dist.loc_coefficients}")
    print(f"\nTrue mu coefficients: {true_mu_coef}")
    print(f"Fitted mu coefficients: {gev_dist.mu_coefficients}")

    print(f"\nNegative log-likelihood: {optResult_gev['neg_log_likelihood']:.4f}")
    print(f"Success: {optResult_gev['success']}")

    # now use a normal dist.
    ndist = scipy.stats.norm
    norm_y = ndist(loc=location, scale=scale).rvs(random_state=42, size=n_samples)
    print("\nFitting Normal with covariates (small demo)...")
    norm_dist,optResult = CovariateDistribution.fit(ndist, norm_y, X_loc=X_loc, X_mu=X_mu, method='Nelder-Mead')
    if norm_dist.shape is not None:
        raise ValueError(f"Expected shape to be None -- is {norm_dist.shape}")

    print("\n=== Results ===")
    print(f"Fitted shape: {norm_dist.shape}")
    print(f"\nTrue location coefficients: {true_loc_coef}")
    print(f"Fitted location coefficients: {norm_dist.loc_coefficients}")
    print(f"\nTrue mu coefficients: {true_mu_coef}")
    print(f"Fitted mu coefficients: {norm_dist.mu_coefficients}")

    print(f"\nNegative log-likelihood: {optResult['neg_log_likelihood']:.4f}")
    print(f"Success: {optResult['success']}")

    # make q-q plot
    qq_plt = gev_dist.qq_pit_plot(gev_y,X_loc,X_mu,num='gev')
    qq_plt['fig'].suptitle('GEV')

    qq_plt_norm = norm_dist.qq_pit_plot(norm_y,X_loc,X_mu,num='norm')
    qq_plt_norm['fig'].suptitle('Normal')

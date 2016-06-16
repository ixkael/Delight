
import numpy as np
import GPy

from GPy.core.model import Model
from GPy.likelihoods.gaussian import HeteroscedasticGaussian
from GPy.inference.latent_function_inference import exact_gaussian_inference
from GPy.core.parameterization.param import Param
from GPy.plotting.gpy_plot.gp_plots import plot
from paramz import ObsAr

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel


class PhotozGP(Model):
    """
    Photo-z Gaussian process, with physical kernel and mean function.
    Default: all parameters are variable except bands and likelihood/noise.
    """
    def __init__(self,
                 redshifts, luminosities, types,
                 noisy_fluxes, flux_variances, bandsUsed,
                 fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 lines_mu, lines_sig,
                 alpha, beta, var_C, var_L,
                 alpha_C, alpha_L, alpha_T,
                 prior_z_t=None,
                 prior_ell_t=None,
                 prior_t=None,
                 X_inducing=None,
                 fix_inducing_to_mean_prediction=True,
                 name='photozgp'):

        super(PhotozGP, self).__init__(name)

        self.nbands, self.numCoefs = fcoefs_amp.shape
        self.num_points, self.numBandsUsed = noisy_fluxes.shape
        assert self.numBandsUsed == len(bandsUsed)
        assert np.min(bandsUsed) >= 0 and np.max(bandsUsed) < self.nbands
        assert fcoefs_amp.shape[0] == self.nbands
        assert fcoefs_mu.shape[0] == self.nbands
        assert fcoefs_sig.shape[0] == self.nbands
        self.bandsUsed = bandsUsed
        self.Y = ObsAr(noisy_fluxes.T.reshape((-1, 1)))
        Ny, self.output_dim = self.Y.shape

        assert redshifts.shape[1] == 1
        assert luminosities.shape[1] == 1
        assert types.shape[1] == 1
        assert self.num_points == redshifts.shape[0] and\
            self.num_points == luminosities.shape[0] and\
            self.num_points == types.shape[0]

        nd = self.num_points
        self.X = np.zeros((nd*self.numBandsUsed, 4))
        for i in range(self.numBandsUsed):
            self.X[i*nd:(i+1)*nd, 0] = bandsUsed[i]
            self.X[i*nd:(i+1)*nd, 1] = redshifts.flatten()
            self.X[i*nd:(i+1)*nd, 2] = luminosities.flatten()
            self.X[i*nd:(i+1)*nd, 3] = types.flatten()

        self.redshifts = Param('redshifts', redshifts)
        self.redshifts.constrain_positive()
        self.link_parameter(self.redshifts)

        self.luminosities = Param('luminosities', luminosities)
        self.luminosities.constrain_positive()
        self.link_parameter(self.luminosities)

        self.types = Param('types', types)
        self.types.constrain_bounded(0, 1)
        self.link_parameter(self.types)

        self.num_data, self.input_dim = self.X.shape

        self.mean_function = Photoz_mean_function(alpha, beta,
                                                  fcoefs_amp,
                                                  fcoefs_mu,
                                                  fcoefs_sig)
        self.link_parameter(self.mean_function)

        self.kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                                  lines_mu, lines_sig, var_C, var_L,
                                  alpha_C, alpha_L, alpha_T)
        self.link_parameter(self.kern)

        self.Y_metadata = {
            'output_index': np.arange(Ny)[:, None],
        }

        self.likelihood = HeteroscedasticGaussian(self.Y_metadata)
        self.likelihood.variance.fix(flux_variances.T.reshape((-1, 1)))

        self.inference_method =\
            exact_gaussian_inference.ExactGaussianInference()

        self.posterior = None

        self.prior_z_t = prior_z_t
        if prior_z_t is not None:
            self.link_parameter(self.prior_z_t)
        self.prior_ell_t = prior_ell_t
        if prior_ell_t is not None:
            self.link_parameter(self.prior_ell_t)
        self.prior_t = prior_t
        if prior_t is not None:
            self.link_parameter(self.prior_t)

        self.X_inducing = None
        if X_inducing is not None:
            assert X_inducing.shape[1] == self.input_dim
            self.X_inducing = X_inducing
            self.Y_inducing = np.zeros((X_inducing.shape[0], 1))
            if not fix_inducing_to_mean_prediction:  # also sample Y_inducing!
                # Otherwise Y_inducing will be set to mean GP prediction
                self.Y_inducing.constrain_positive()
                self.link_parameter(self.Y_inducing)

    def set_unfixed_parameters(self, params):
        """Set unfixed parameters all at once"""
        for i, nm in enumerate(self.parameter_names_flat()):
            self[nm[9:]] = params[i]
            # TODO: make sure no updates are triggered before!
        self.parameters_changed()

    def set_redshifts(self, redshifts):
        """Set redshifts"""
        assert redshifts.shape[1] == 1
        assert redshifts.shape[0] == self.types.shape[0] and\
            redshifts.shape[0] == self.luminosities.shape[0]
        self.update_model(False)
        index = self.redshifts._parent_index_
        self.unlink_parameter(self.redshifts)
        self.redshifts = Param('redshifts', redshifts)
        self.redshifts.constrain_positive()
        self.link_parameter(self.redshifts, index=index)
        self.update_model(True)

    def set_luminosities(self, luminosities):
        """Set luminosities"""
        assert luminosities.shape[1] == 1
        assert luminosities.shape[0] == self.types.shape[0] and\
            luminosities.shape[0] == self.redshifts.shape[0]
        self.update_model(False)
        index = self.luminosities._parent_index_
        self.unlink_parameter(self.luminosities)
        self.luminosities = Param('luminosities', luminosities)
        self.luminosities.constrain_positive()
        self.link_parameter(self.luminosities, index=index)
        self.update_model(True)

    def set_types(self, types):
        """Set types"""
        assert types.shape[1] == 1
        assert types.shape[0] == self.redshifts.shape[0] and\
            types.shape[0] == self.luminosities.shape[0]
        self.update_model(False)
        index = self.types._parent_index_
        self.unlink_parameter(self.types)
        self.types = Param('types', types)
        self.types.constrain_bounded(0, 1)
        self.link_parameter(self.types, index=index)
        self.update_model(True)

    def parameters_changed(self):
        """If parameters changed, compute gradients"""
        nd = self.num_points
        self.X = np.zeros((nd*self.numBandsUsed, 4))
        for i in range(self.numBandsUsed):
            self.X[i*nd:(i+1)*nd, 0] = self.bandsUsed[i]
            self.X[i*nd:(i+1)*nd, 1] = self.redshifts.values.flatten()
            self.X[i*nd:(i+1)*nd, 2] = self.luminosities.values.flatten()
            self.X[i*nd:(i+1)*nd, 3] = self.types.values.flatten()

        self.posterior, self._log_marginal_likelihood, self.grad_dict\
            = self.inference_method.inference(self.kern, self.X,
                                              self.likelihood,
                                              self.Y, self.mean_function,
                                              self.Y_metadata)

        if self.X_inducing is not None:
            if fix_inducing_to_mean_prediction:
                mu, var = self._raw_predict(X_inducing, full_cov=False)
                self.Y_inducing = mu
            else:
                self._log_marginal_likelihood +=\
                    self.log_predictive_density(X_inducing, Y_inducing)
                raise NotImplementedError("Uncertain inducing not implemented")
                #  TODO : update gradients of inducting points
                #  self.Y_inducing.gradient =

        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])

        self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)

        wm = self.mean_function.gradients_X(self.grad_dict['dL_dm'], self.X)
        wk = self.kern.gradients_X_diag(self.grad_dict['dL_dK'], self.X)
        gradX = wm + wk

        if not self.redshifts.is_fixed:
            self.redshifts.gradient[:] = 0
        if not self.luminosities.is_fixed:
            self.luminosities.gradient[:] = 0
        if not self.types.is_fixed:
            self.types.gradient[:] = 0

        if not self.redshifts.is_fixed:
            for i in range(self.numBandsUsed):
                self.redshifts.gradient[:, 0] += gradX[i*nd:(i+1)*nd, 1]
            if self.prior_z_t is not None:
                self._log_marginal_likelihood += np.sum(
                    self.prior_z_t.lnpdf(self.redshifts, self.types))
                self.prior_z_t.update_gradients(1, self.redshifts, self.types)
                self.redshifts.gradient +=\
                    self.prior_z_t.lnpdf_grad_z(self.redshifts, self.types)
                if not self.types.is_fixed:
                    self.types.gradient +=\
                        self.prior_z_t.lnpdf_grad_t(self.redshifts, self.types)

        if not self.luminosities.is_fixed:
            for i in range(self.numBandsUsed):
                self.luminosities.gradient[:, 0] += gradX[i*nd:(i+1)*nd, 2]
            if self.prior_ell_t is not None:
                self._log_marginal_likelihood += np.sum(
                    self.prior_ell_t.lnpdf(self.luminosities, self.types))
                self.prior_ell_t.update_gradients(1, self.luminosities,
                                                  self.types)
                self.luminosities.gradient +=\
                    self.prior_ell_t.lnpdf_grad_ell(self.luminosities,
                                                    self.types)
                if not self.types.is_fixed:
                    self.types.gradient +=\
                        self.prior_ell_t.lnpdf_grad_t(self.luminosities,
                                                      self.types)

        if not self.types.is_fixed:
            for i in range(self.numBandsUsed):
                self.types.gradient[:, 0] += gradX[i*nd:(i+1)*nd, 3]
            if self.prior_t is not None:
                self._log_marginal_likelihood += np.sum(
                    self.prior_t.lnpdf(self.types))
                self.prior_t.update_gradients(1, self.types)
                self.types.gradient +=\
                    self.prior_t.lnpdf_grad_t(self.types)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        For making predictions, without normalization or likelihood
        """
        mu, var = self.posterior._raw_predict(
            kern=self.kern if kern is None else kern,
            Xnew=Xnew, pred_var=self.X, full_cov=full_cov)
        mu += self.mean_function.f(Xnew)
        return mu, var

    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        """
        Calculation of the log predictive density
        """
        mu_star, var_star = self._raw_predict(x_test)
        return self.likelihood.log_predictive_density(y_test, mu_star,
                                                      var_star,
                                                      Y_metadata=Y_metadata)

    def plot_f(self, plot_limits=None, fixed_inputs=None,
               resolution=None,
               apply_link=False,
               which_data_ycols='all', which_data_rows='all',
               visible_dims=None,
               levels=20, samples=0, lower=2.5, upper=97.5,
               plot_density=False,
               plot_data=True, plot_inducing=True,
               projection='2d', legend=True,
               predict_kw=None,
               **kwargs):
        """
        Convenience function for plotting the fit of a GP.
        """
        return plot(self, plot_limits, fixed_inputs, resolution, True,
                    apply_link, which_data_ycols, which_data_rows,
                    visible_dims, levels, samples, 0,
                    lower, upper, plot_data, plot_inducing,
                    plot_density, predict_kw, projection, legend, **kwargs)

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None,
                likelihood=None, include_likelihood=True):
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)
        if include_likelihood:
            if likelihood is None:
                likelihood = self.likelihood
            mu, var = likelihood.predictive_values(mu, var,
                                                   full_cov,
                                                   Y_metadata=Y_metadata)
        return mu, var

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None,
                          kern=None, likelihood=None):
        """
        Get the predictive quantiles around the prediction at X
        """
        m, v = self._raw_predict(X,  full_cov=False, kern=kern)
        if likelihood is None:
            likelihood = self.likelihood
        return likelihood.predictive_quantiles(m, v, quantiles,
                                               Y_metadata=Y_metadata)

    def posterior_samples_f(self, X, size=10, full_cov=True, **predict_kwargs):
        """
        Samples the posterior GP at the points X.
        """
        m, v = self._raw_predict(X,  full_cov=full_cov, **predict_kwargs)

        def sim_one_dim(m, v):
            if not full_cov:
                return np.random.multivariate_normal(m.flatten(),
                                                     np.diag(v.flatten()),
                                                     size).T
            else:
                return np.random.multivariate_normal(m.flatten(), v, size).T

        if self.output_dim == 1:
            return sim_one_dim(m, v)
        else:
            fsim = np.empty((self.output_dim, self.num_data, size))
            for d in range(self.output_dim):
                if full_cov and v.ndim == 3:
                    fsim[d] = sim_one_dim(m[:, d], v[:, :, d])
                elif (not full_cov) and v.ndim == 2:
                    fsim[d] = sim_one_dim(m[:, d], v[:, d])
                else:
                    fsim[d] = sim_one_dim(m[:, d], v)
        return fsim

    def posterior_samples(self, X, size=10, full_cov=False, Y_metadata=None,
                          likelihood=None, **predict_kwargs):
        """
        Samples the posterior GP at the points X.
        """
        fsim = self.posterior_samples_f(X, size, full_cov=full_cov,
                                        **predict_kwargs)
        if likelihood is None:
            likelihood = self.likelihood
        if fsim.ndim == 3:
            for d in range(fsim.shape[0]):
                fsim[d] = likelihood.samples(fsim[d], Y_metadata=Y_metadata)
        else:
            fsim = likelihood.samples(fsim, Y_metadata=Y_metadata)
        return fsim

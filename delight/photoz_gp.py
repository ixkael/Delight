
import numpy as np
import GPy

from GPy.core.model import Model
from GPy.likelihoods.gaussian import HeteroscedasticGaussian
from GPy.inference.latent_function_inference import exact_gaussian_inference
from GPy.util.linalg import pdinv, dpotrs, tdot, dtrtrs
from GPy.core.parameterization.param import Param
from GPy.plotting.gpy_plot.gp_plots import plot
from paramz import ObsAr
import re
from copy import copy

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
# TODO: add tests for prediction routines

log_2_pi = np.log(2*np.pi)


class PhotozGP(Model):
    """
    Photo-z Gaussian process, with physical kernel and mean function.
    Default: all parameters are variable except bands and likelihood/noise.
    """
    def __init__(self,
                 redshifts, luminosities, types, unfixed_indices,
                 noisy_fluxes, flux_variances, extranoise, bandsUsed,
                 fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 lines_mu, lines_sig,
                 var_C, var_L,
                 alpha_C, alpha_L, alpha_T,
                 prior_z_t=None,
                 prior_ell_t=None,
                 prior_t=None,
                 X_inducing=None,
                 redshiftGrid=None,
                 use_interpolators=True,
                 name='photozgp'):

        super(PhotozGP, self).__init__(name)

        self.use_interpolators = use_interpolators
        assert flux_variances.shape == noisy_fluxes.shape
        self.nbands, self.numCoefs = fcoefs_amp.shape
        self.num_points, self.numBandsUsed = noisy_fluxes.shape
        assert self.numBandsUsed == len(bandsUsed)
        assert np.min(bandsUsed) >= 0 and np.max(bandsUsed) < self.nbands
        assert fcoefs_amp.shape[0] == self.nbands
        assert fcoefs_mu.shape[0] == self.nbands
        assert fcoefs_sig.shape[0] == self.nbands
        self.bandsUsed = copy(bandsUsed)
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

        assert np.min(unfixed_indices) >= 0 and np.max(unfixed_indices) < nd
        self.unfixed_indices = copy(unfixed_indices)

        self.redshifts = copy(redshifts)
        self.unfixed_redshifts\
            = Param('redshifts', copy(redshifts[unfixed_indices, :]))
        self.unfixed_redshifts.constrain_positive()
        self.link_parameter(self.unfixed_redshifts)

        self.luminosities = copy(luminosities)
        self.unfixed_luminosities\
            = Param('luminosities', copy(luminosities[unfixed_indices, :]))
        self.unfixed_luminosities.constrain_positive()
        self.link_parameter(self.unfixed_luminosities)

        self.types = copy(types)
        self.unfixed_types\
            = Param('types', copy(types[unfixed_indices, :]))
        self.unfixed_types.constrain_bounded(0, 1)
        self.link_parameter(self.unfixed_types)

        self.num_data, self.input_dim = self.X.shape

        self.mean_function = Photoz_mean_function(fcoefs_amp,
                                                  fcoefs_mu,
                                                  fcoefs_sig)
        self.link_parameter(self.mean_function)

        self.kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                                  lines_mu, lines_sig, var_C, var_L,
                                  alpha_C, alpha_L, alpha_T,
                                  redshiftGrid=redshiftGrid,
                                  use_interpolators=self.use_interpolators)
        self.link_parameter(self.kern)

        self.Y_metadata = {
            'output_index': np.arange(Ny)[:, None],
        }

        self.extranoise = Param('extranoise', float(extranoise))
        self.extranoise.constrain_positive()
        self.link_parameter(self.extranoise)
        self.likelihood = HeteroscedasticGaussian(self.Y_metadata)
        self.flux_variances = flux_variances.T.reshape((-1, 1))
        self.likelihood.variance.fix(self.flux_variances + self.extranoise)

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

        self.X_inducing = X_inducing
        self.derived_params = []
        self.derived_param_names = []
        if self.X_inducing is not None:
            assert self.X_inducing.shape[1] == self.input_dim
            self.Y_inducing_mean = np.zeros((self.X_inducing.shape[0], 1))
            self.Y_inducing_std = np.zeros((self.X_inducing.shape[0], 1))
            self.derived_params.append(self.Y_inducing_mean)
            self.derived_param_names.append('Y_inducing_mean')
            self.derived_params.append(self.Y_inducing_std)
            self.derived_param_names.append('Y_inducing_std')

    def get_param_names_and_indices(self):
        scalar_params, array_params = {}, {}
        for i, nm in enumerate(self.parameter_names_flat()):
            nmb = nm[9:]
            if '[[' in nmb and ']]' in nmb:
                i1, i2 = np.array(re.findall("\[\[(.*?)\]\]", nmb)[0]
                                  .split(), dtype=int)
                nmb = nmb.split('[[')[0]
                if nmb not in array_params:
                    array_params[nmb] = ([[i1, i2]], [i])
                else:
                    idx1, idx2 = array_params[nmb]
                    idx1.append([i1, i2])
                    idx2.append(i)
                    array_params[nmb] = (idx1, idx2)
            else:
                scalar_params[nmb] = i
        for nmb, (idx1, idx2) in array_params.iteritems():
            array_params[nmb] = (np.array(idx1), np.array(idx2))
        return scalar_params, array_params

    def set_unfixed_parameters(self, params, scalar_params, array_params):
        """Set unfixed parameters all at once"""
        self.update_model(False)
        for nmb, (idx1, idx2) in array_params.iteritems():
            self[nmb][idx1[:, 0], idx1[:, 1]] = params[idx2]
        for nmb, i in scalar_params.iteritems():
            self[nmb] = params[i]
        self.update_model(True)

    def set_extranoise(self, extranoise):
        self.update_model(False)
        index = self.extranoise._parent_index_
        self.unlink_parameter(self.extranoise)
        self.extranoise = Param('extranoise', float(extranoise))
        self.extranoise.constrain_positive()
        self.link_parameter(self.extranoise, index=index)
        self.update_model(True)

    def set_redshifts(self, redshifts):
        """Set redshifts"""
        assert redshifts.shape[1] == 1
        assert redshifts.shape[0] == self.types.shape[0] and\
            redshifts.shape[0] == self.luminosities.shape[0]
        self.update_model(False)
        self.redshifts = copy(redshifts)
        self.unfixed_redshifts.values[:, 0]\
            = redshifts[self.unfixed_indices, 0]
        self.update_model(True)

    def set_luminosities(self, luminosities):
        """Set luminosities"""
        assert luminosities.shape[1] == 1
        assert luminosities.shape[0] == self.types.shape[0] and\
            luminosities.shape[0] == self.redshifts.shape[0]
        self.update_model(False)
        self.luminosities = copy(luminosities)
        self.unfixed_luminosities.values[:, 0]\
            = luminosities[self.unfixed_indices, 0]
        self.update_model(True)

    def set_types(self, types):
        """Set types"""
        assert types.shape[1] == 1
        assert types.shape[0] == self.redshifts.shape[0] and\
            types.shape[0] == self.luminosities.shape[0]
        self.update_model(False)
        self.types = copy(types)
        self.unfixed_types.values[:, 0]\
            = types[self.unfixed_indices, 0]
        self.update_model(True)

    def parameters_changed(self):
        """If parameters changed, compute gradients"""
        nd = self.num_points
        self.redshifts[self.unfixed_indices, :]\
            = self.unfixed_redshifts.values
        self.types[self.unfixed_indices, :]\
            = self.unfixed_types.values
        self.luminosities[self.unfixed_indices, :]\
            = self.unfixed_luminosities.values
        self.X = np.zeros((nd*self.numBandsUsed, 4))
        for i in range(self.numBandsUsed):
            self.X[i*nd:(i+1)*nd, 0] = self.bandsUsed[i]
            self.X[i*nd:(i+1)*nd, 1] = self.redshifts.flatten()
            self.X[i*nd:(i+1)*nd, 2] = self.luminosities.flatten()
            self.X[i*nd:(i+1)*nd, 3] = self.types.flatten()

        self.posterior, self._log_marginal_likelihood, self.grad_dict\
            = self.inference_method.inference(self.kern, self.X,
                                              self.likelihood,
                                              self.Y, self.mean_function,
                                              self.Y_metadata)

        self.likelihood.variance.fix(self.flux_variances + self.extranoise)
        self.extranoise.gradient\
            = self.grad_dict['dL_dthetaL'].sum()
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])

        self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

        gradX = self.mean_function.gradients_X(self.grad_dict['dL_dm'], self.X)

        var_C_gradient_X, var_L_gradient_X, alpha_C_gradient_X,\
            alpha_L_gradient_X, alpha_T_gradient_X\
            = self.kern.get_gradients_full(self.X)

        grad_ell_X, grad_t_X = self.kern.get_gradients_X(self.X)

        dL_dK = self.grad_dict['dL_dK']
        self.kern.var_C.gradient = np.sum(dL_dK * var_C_gradient_X)
        self.kern.var_L.gradient = np.sum(dL_dK * var_L_gradient_X)
        self.kern.alpha_C.gradient = np.sum(dL_dK * alpha_C_gradient_X)
        self.kern.alpha_T.gradient = np.sum(dL_dK * alpha_T_gradient_X)
        self.kern.alpha_L.gradient = np.sum(dL_dK * alpha_L_gradient_X)

        gradX[:, 2] += np.sum(dL_dK * grad_ell_X, axis=1)
        gradX[:, 3] += np.sum(dL_dK * grad_t_X, axis=1)

        if self.X_inducing is not None:

            mu, var, marglike, dL_dKz, V = self._raw_predict(
                self.X_inducing, full_cov=True, marglike=True)
            self.Y_inducing_mean[:, 0] = mu[:, 0]
            self.Y_inducing_std[:, 0] = np.sqrt(np.diag(var))

            if False:
                self._log_marginal_likelihood += marglike

                var_C_gradient_Z, var_L_gradient_Z, alpha_C_gradient_Z,\
                    alpha_L_gradient_Z, alpha_T_gradient_Z\
                    = self.kern.get_gradients_full(self.X_inducing)

                var_C_gradient_XZ, var_L_gradient_XZ, alpha_C_gradient_XZ,\
                    alpha_L_gradient_XZ, alpha_T_gradient_XZ\
                    = self.kern.get_gradients_full(self.X, self.X_inducing)

                grad_ell_XZ, grad_t_XZ = self.kern.get_gradients_X(
                    self.X, self.X_inducing)

                var_C_gradient = var_C_gradient_Z\
                    + np.dot(np.dot(V.T, var_C_gradient_X), V)\
                    - 2*np.dot(V.T, var_C_gradient_XZ)
                var_L_gradient = var_L_gradient_Z\
                    + np.dot(np.dot(V.T, var_L_gradient_X), V)\
                    - np.dot(V.T, var_L_gradient_XZ)
                alpha_C_gradient = alpha_C_gradient_Z\
                    + np.dot(np.dot(V.T, alpha_C_gradient_X), V)\
                    - 2*np.dot(V.T, alpha_C_gradient_XZ)
                alpha_T_gradient = alpha_T_gradient_Z\
                    + np.dot(np.dot(V.T, alpha_T_gradient_X), V)\
                    - 2*np.dot(V.T, alpha_T_gradient_XZ)
                alpha_L_gradient = alpha_L_gradient_Z\
                    + np.dot(np.dot(V.T, alpha_L_gradient_X), V)\
                    - 2*np.dot(V.T, alpha_L_gradient_XZ)
                self.kern.var_C.gradient += np.sum(dL_dKz * var_C_gradient)
                self.kern.var_L.gradient += np.sum(dL_dKz * var_L_gradient)
                self.kern.alpha_C.gradient += np.sum(dL_dKz * alpha_C_gradient)
                self.kern.alpha_T.gradient += np.sum(dL_dKz * alpha_T_gradient)
                self.kern.alpha_L.gradient += np.sum(dL_dKz * alpha_L_gradient)

                #  TODO: implement gradients_X w.r.t. inducing marg like
                #  grad_ell =  - 2 * np.dot(V.T, grad_ell_XZ)
                #  grad_t = - 2 * np.dot(V.T, grad_t_XZ)
                #  gradX[:, 2] += np.sum(dL_dKz * grad_ell, axis=1)
                #  gradX[:, 3] += np.sum(dL_dKz * grad_t, axis=1)

        if not self.unfixed_redshifts.is_fixed:
            self.unfixed_redshifts.gradient[:] = 0
        if not self.unfixed_luminosities.is_fixed:
            self.unfixed_luminosities.gradient[:] = 0
        if not self.unfixed_types.is_fixed:
            self.unfixed_types.gradient[:] = 0

        if not self.unfixed_redshifts.is_fixed:
            for i in range(self.numBandsUsed):
                self.unfixed_redshifts.gradient[:, 0]\
                    += gradX[i*nd:(i+1)*nd, 1][self.unfixed_indices]
            if self.prior_z_t is not None:
                self._log_marginal_likelihood += np.sum(
                    self.prior_z_t.lnpdf(self.redshifts,
                                         self.types))
                self.prior_z_t.update_gradients(1, self.redshifts,
                                                self.types)
                self.unfixed_redshifts.gradient +=\
                    self.prior_z_t.lnpdf_grad_z(self.unfixed_redshifts,
                                                self.unfixed_types)
                if not self.unfixed_types.is_fixed:
                    self.unfixed_types.gradient +=\
                        self.prior_z_t.lnpdf_grad_t(self.unfixed_redshifts,
                                                    self.unfixed_types)

        if not self.unfixed_luminosities.is_fixed:
            for i in range(self.numBandsUsed):
                self.unfixed_luminosities.gradient[:, 0]\
                    += gradX[i*nd:(i+1)*nd, 2][self.unfixed_indices]
            if self.prior_ell_t is not None:
                self._log_marginal_likelihood += np.sum(
                    self.prior_ell_t.lnpdf(self.luminosities,
                                           self.types))
                self.prior_ell_t.update_gradients(1, self.luminosities,
                                                  self.types)
                self.unfixed_luminosities.gradient +=\
                    self.prior_ell_t.lnpdf_grad_ell(self.unfixed_luminosities,
                                                    self.unfixed_types)
                if not self.unfixed_types.is_fixed:
                    self.unfixed_types.gradient +=\
                        self.prior_ell_t.lnpdf_grad_t(
                            self.unfixed_luminosities,
                            self.unfixed_types)

        if not self.unfixed_types.is_fixed:
            for i in range(self.numBandsUsed):
                self.unfixed_types.gradient[:, 0]\
                    += gradX[i*nd:(i+1)*nd, 3][self.unfixed_indices]
            if self.prior_t is not None:
                self._log_marginal_likelihood += np.sum(
                    self.prior_t.lnpdf(self.types))
                self.prior_t.update_gradients(1, self.types)
                self.unfixed_types.gradient +=\
                    self.prior_t.lnpdf_grad_t(self.unfixed_types)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _raw_predict(self, Xnew, full_cov=False, marglike=False):
        kern = self.kern
        Kx = kern.K(self.X, Xnew)
        mu = np.dot(Kx.T, self.posterior.woodbury_vector)
        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)
        if full_cov:
            Kxx = kern.K(Xnew)
            if self.posterior._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self.posterior._woodbury_chol, Kx)[0]
                var = Kxx - tdot(tmp.T)
            elif self.posterior._woodbury_chol.ndim == 3:  # Missing data
                var = np.empty((Kxx.shape[0],
                                Kxx.shape[1],
                                self.posterior._woodbury_chol.shape[2]))
                for i in range(var.shape[2]):
                    tmp = dtrtrs(self.posterior._woodbury_chol[:, :, i], Kx)[0]
                    var[:, :, i] = (Kxx - tdot(tmp.T))
            var = var
        else:
            Kxx = kern.Kdiag(Xnew)
            if self.posterior._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = (Kxx - np.square(tmp).sum(0))[:, None]
            elif self.posterior._woodbury_chol.ndim == 3:  # Missing data
                var = np.empty((Kxx.shape[0],
                                self.posterior._woodbury_chol.shape[2]))
                for i in range(var.shape[1]):
                    tmp = dtrtrs(self.posterior._woodbury_chol[:, :, i], Kx)[0]
                    var[:, i] = (Kxx - np.square(tmp).sum(0))
            var = var
        mu += self.mean_function.f(Xnew)
        if marglike is True and full_cov is True:
            Wi, LW, LWi, W_logdet = pdinv(var)
            log_marginal = 0.5*(-mu.size * log_2_pi -
                                mu.shape[1] * W_logdet)
            dL_dKnn = - 0.5 * mu.shape[1] * Wi
            V = np.dot(self.posterior.woodbury_inv, Kx)
            return mu, var, log_marginal, dL_dKnn, V
        else:
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

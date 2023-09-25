import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, transforms
import jax.numpy as jnp


def plot_ellipse(Sigma, mu, ax, n_std=3.0, facecolor="none", edgecolor="k", **kwargs):
    """Plot an ellipse to with centre `mu` and axes defined by `Sigma`."""
    cov = Sigma
    pearson = cov[0, 1] / jnp.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = jnp.sqrt(1 + pearson)
    ell_radius_y = jnp.sqrt(1 - pearson)

    # if facecolor not in kwargs:
    #     kwargs['facecolor'] = 'none'
    # if edgecolor not in kwargs:
    #     kwargs['edgecolor'] = 'k'

    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, edgecolor=edgecolor, **kwargs
    )

    scale_x = jnp.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = jnp.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_uncertainty_ellipses(means, Sigmas, ax, n_std=3.0, **kwargs):
    """Loop over means and Sigmas to add ellipses representing uncertainty."""
    for Sigma, mu in zip(Sigmas, means):
        plot_ellipse(Sigma, mu, ax, n_std, **kwargs)


def plot_posterior_covariance(post_means, post_covs, ax=None, ellipse_kwargs={}, legend_kwargs={}, **kwargs):
    """Plot posterior means and covariances for the first two dimensions of
     the latent state of a LGSSM.

    Args:
        post_means: array(T, D).
        post_covs: array(T, D, D).
        ax: matplotlib axis.
        ellipse_kwargs: keyword arguments passed to matplotlib.patches.Ellipse().
        **kwargs: passed to ax.plot().
    """
    if ax is None:
        fig, ax = plt.subplots()

    # This is to stop some weird behaviour where running the function multiple
    # #  times with an empty argument wouldn't reset the dictionary.
    # if ellipse_kwargs is None:
    #     ellipse_kwargs = dict()

    # if 'edgecolor' not in ellipse_kwargs:
    #     if 'color' in kwargs:
    #         ellipse_kwargs['edgecolor'] = kwargs['color']

    # Select the first two dimensions of the latent space.
    # post_means = post_means[:, :2]
    # post_covs = post_covs[:, :2, :2]

    # Plot the mean trajectory
    # ax.plot(post_means[:, 0], post_means[:, 1], **kwargs)
    # Plot covariance at each time point.
    plot_uncertainty_ellipses(post_means, post_covs, ax, **ellipse_kwargs)

    ax.axis("equal")

    if "label" in kwargs:
        ax.legend(**legend_kwargs)

    return ax
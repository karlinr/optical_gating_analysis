import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def radsperframe_to_bps(radsperframe, framerate):
    return (radsperframe * framerate) / (2 * np.pi)

def plot_oog_general(phases, delta_phases, drifts, frame_rate):
    plt.figure(figsize = (16,16))
    plt.subplot(221)
    plt.title("Phases")
    plt.plot(phases)
    plt.xlabel("Frame number")
    plt.ylabel("Phase (rad)")

    plt.subplot(222)
    plt.title("Phases against delta-phases")
    plt.scatter(phases[1::],radsperframe_to_bps(delta_phases, frame_rate), s = 5)
    plt.xlabel("Phase (rad)")
    plt.ylabel("Beat velocity (beats/s)")

    plt.subplot(223)
    plt.title("Drifts")
    plt.plot(np.array(drifts)[:,0], label = "x")
    plt.plot(np.array(drifts)[:,1], label = "y")
    plt.legend()
    plt.xlabel("Frame number")
    plt.ylabel("Drift (pixels)")

    plt.subplot(224)
    plt.title("Delta-phases against frame number")
    plt.scatter(range(delta_phases.shape[0]), radsperframe_to_bps(delta_phases, frame_rate), s = 5)
    plt.xlabel("Frame number")
    plt.ylabel("Beat velocity (beats/s)")

    plt.show()

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_poincare(gradients, frame_rate):
    plt.figure(figsize=(9,9))
    plt.scatter(radsperframe_to_bps(np.array(gradients[:-1]), frame_rate), radsperframe_to_bps(np.array(gradients[1:]), frame_rate), s = 20, c = range(len(gradients) - 1))
    ax = plt.gca()
    confidence_ellipse(radsperframe_to_bps(np.array(gradients[:-1]), frame_rate), radsperframe_to_bps(np.array(gradients[1:]), frame_rate), ax, edgecolor='red')
    plt.colorbar(label = "Beat index")
    plt.xlabel("Beat velocity (beats/s), $f_n$")
    plt.ylabel("Beat velocity (beats/s), $f_{n+1}$")
    plt.axis('equal')
    plt.show()
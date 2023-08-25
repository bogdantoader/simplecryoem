import numpy as np
import jax.numpy as jnp
from simplecryoem.utils import crop_fourier_images


def estimate_noise(imgs, nx_empty=48, nx_final=32):
    """Givben an array [N, nx0, nx0] of real centred images, estimate the
    pixel-wise Fourier noise using the empty corners of the real images.

    Parameters:
    ----------
    imgs : [N, nx0, nx0] array
        The images for which we want to estimate the noise.

    nx_empty : int
        The side length of the corner to use for estimating the noise.

    nx_final: int
        The dimension of the cropped images (and of the noise estimation).

    Returns:
    -------

    stddev : [nx_final * nx_final] int array
        The standard deviation of the Fourier coefficients of the noise,
        with the standard ordering and reshaped as a 1D array.

    """

    nx0 = imgs.shape[2]

    # Crop the empty corners from the real images.
    corners = imgs[:, :nx_empty, :nx_empty]

    # Take padded FFT so that the result has the same dimensions as the
    # initial images. If need to do on many images, apply fft2 to smaller
    # bathces of images.
    f_corners = np.fft.fft2(corners, s=[nx0, nx0])

    # Crop the FFT of the empty corner in the same way that we will crop
    # the particle images
    x_grid = [1, f_corners.shape[2]]
    f_corners, _ = crop_fourier_images(f_corners, x_grid, nx_final)

    # Now we have the Fourier transforms of the noise, of the same
    # dimensions and crop as the particle images.
    # Compute the standard deviation of the noise, with the appropriate
    # scaling due to taking Fourier transforms
    # (scaling empirically determined, to check on paper).
    stddev = np.std(f_corners, axis=0) / nx_empty * nx0

    return stddev.reshape(-1)


def average_radially(img, x_grid):
    """Radially average a 2D array in the Fourier domain."""

    x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1 / (x_grid[0] * x_grid[1]))
    X, Y = jnp.meshgrid(x_freq, x_freq)
    r = jnp.sqrt(X**2 + Y**2)
    rads = jnp.diag(r)
    rads = rads[: jnp.argmax(rads) + 1]
    eps = rads[1] / 2

    img_avg = np.zeros(img.shape)
    for rad_i in rads:
        idx = np.array(jnp.abs(r - rad_i) <= eps)
        img_rad_avg = jnp.mean(img[idx])
        img_avg[idx] = img_rad_avg

    return jnp.array(img_avg)


def estimate_noise_radial(imgs, nx_empty=48, nx_final=32):
    """Wrapper around estimate_noise_imgs and radial averaging
    of the output."""

    print("Estimating pixel-wise noise...", end="", flush=True)
    sigma_noise = estimate_noise(imgs, nx_empty, nx_final).reshape([nx_final, nx_final])
    print("done.")

    x_grid = [1, sigma_noise.shape[0]]

    print("Averaging radially...", end="", flush=True)
    sigma_noise_avg = average_radially(sigma_noise, x_grid)
    print("done.")

    return sigma_noise_avg.reshape(-1)

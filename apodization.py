import numpy as np
from matplotlib import pyplot as plt


def rayleigh_range(w0, wavelength):
    """
    Rayleigh range z_R for a Gaussian beam.

    Parameters
    ----------
    w0 : float [m]        Waist radius at focus (1/e^2)
    wavelength : float [m]

    Returns
    -------
    z_R : float [m]
    """
    return np.pi * w0**2 / wavelength


def beam_radius_from_oap(s, f, w0, wavelength):
    """
    1/e^2 beam radius w(s) at distance s from the OAP mirror.

    Coordinate: s = 0 at the OAP; s = f at the focal waist.

    Parameters
    ----------
    s : float or array [m]        Axial distance from OAP along propagation
    f : float [m]                 OAP focal length
    w0 : float [m]                Waist radius at the focus (1/e^2)
    wavelength : float [m]        Center wavelength

    Returns
    -------
    w : float or array [m]        Beam radius at position s
    z_R : float [m]               Rayleigh range
    """
    z_R = rayleigh_range(w0, wavelength)
    z_rel = s - f                   # distance from the waist plane
    w = w0 * np.sqrt(1 + (z_rel / z_R)**2)
    return w, z_R


def gaussian_aperture_throughput(aperture_radius, w):
    """
    Power fraction of a circular Gaussian (1/e^2 radius = w) passing a circular
    aperture of radius 'aperture_radius' at that plane.

    Parameters
    ----------
    aperture_radius : float [m]
    w : float or array [m]        1/e^2 radius at the aperture plane

    Returns
    -------
    eta : float or array          Enclosed power fraction (0..1)
    """
    T = aperture_radius / w
    return 1 - np.exp(-2 * T**2)


# --- Example with your numbers ---
if __name__ == "__main__":
    inch = 0.0254
    f = 6 * inch                  # 6 inches
    D = 2 * inch                  # 2-inch OAP diameter
    a = D / 2                     # aperture radius
    w0 = 2e-3                     # 2 mm waist at focus

    # choose a representative THz wavelength (e.g., 1 THz -> 300 µm)
    wavelength = 300e-6

    # beam radius AT the mirror plane (s = 0)
    w_mirror, zR = beam_radius_from_oap(0.0, f, w0, wavelength)

    # check clipping at the OAP
    eta = gaussian_aperture_throughput(a, w_mirror)

    print(f"Rayleigh range zR = {zR*1e3:.2f} mm")
    print(f"Beam radius at OAP (s=0): w = {w_mirror*1e3:.2f} mm")
    print(f"OAP radius = {a*1e3:.2f} mm; enclosed power at mirror = {eta*100:.6f}%")

    z = np.linspace(0, 12*inch, 500)
    w_z, _ = beam_radius_from_oap(z, f, w0, wavelength)
    eta_z = gaussian_aperture_throughput(a, w_z)

    fig, axes = plt.subplots(nrows=2,sharex=True, figsize=(12, 8))

    axes[0].plot(z / inch, w_z / inch, 'k')
    axes[0].set_xlabel('Distance from OAP [inch]')
    axes[0].set_ylabel('Beam radius w [inch]')

    w_0, _ = beam_radius_from_oap(f, f, w0, wavelength)

    axes[1].plot(z / inch, w_0 / w_z, 'k')
    axes[1].set_xlabel('Distance from OAP [inch]')
    axes[1].set_ylabel('Window')

    plt.savefig("beam_radius.png")
    plt.show()
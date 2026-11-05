import numpy as np


def get_rho(v, m, m_err):
    rho = m / v
    rho_err = m_err / v
    return rho, rho_err

if __name__ == "__main__":

    mass_solid_cube = 0.84825
    mass_solid_cube_err = 0.03229422

    mass_spipab_cube = 0.432333333
    mass_spipab_cube_err = 0.010066446

    mass_solid_sphere = 6.52125
    mass_solid_sphere_err = 0.117008191

    mass_spipab_sphere = 3.7545
    mass_spipab_sphere_err = 0.146654924

    cube_vol = 1.0 * 1.0 * 1.0  # cm^3
    r = 2.4 / 2.0
    sphere_vol = (4/3) * np.pi * (r ** 3)  # cm^3

    print(f"Solid cube density: {get_rho(cube_vol, mass_solid_cube, mass_solid_cube_err)} g/cm^3")
    print(f"SPIPA-B cube density: {get_rho(cube_vol, mass_spipab_cube, mass_spipab_cube_err)} g/cm^3")
    print(f"Solid sphere density: {get_rho(sphere_vol, mass_solid_sphere, mass_solid_sphere_err)} g/cm^3")
    print(f"SPIPA-B sphere density: {get_rho(sphere_vol, mass_spipab_sphere, mass_spipab_sphere_err)} g/cm^3")


import os
import numpy as np


# I'm using this package to sample the quasigeoid undulation raster live in the file
# GEOTIFF sourced from 
# https://gdz.bkg.bund.de/index.php/default/digitale-geodaten/geodaetische-basisdaten/quasigeoid-der-bundesrepublik-deutschland-quasigeoid.html
import rasterio

GCG2016_FILE = 'GCG2016v2023.tif'
GCG2016_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), GCG2016_FILE)

class Ellipsoid:
    def __init__(self, a, f, Z = 32):
        self.a = a
        self.f = f
        self.b = (1.0 - f) * a
        self.e_fst_sq = (a**2 - self.b**2) / a**2
        self.e_snd_sq = (a**2 - self.b**2) / self.b**2
        self.r_approx = (self.a * 2.0 + self.b) / 3.0

        assert Z > 0 and Z <= 60
        self.Z = Z

        assert os.path.exists(GCG2016_PATH)
        self.gcg = None

    def N(self, phi):
        return self.a / np.sqrt(1.0 - self.e_fst_sq * np.sin(phi)**2)

    # pos: [x, y, z] in CRS
    def cartesian_to_ellipsoidal(self, pos, eps = 1e-4):
        lam = np.arctan(pos[1] / pos[0])

        p = np.sqrt(pos[0]**2 + pos[1]**2)

        alt = [None, 0.0]
        phi = [None, np.arctan(pos[2] * (1.0 + self.e_snd_sq) / p)]

        cond = lambda v, s: (np.abs(np.diff(v)[0]) * s > eps)
        while (alt[0] is None or phi[0] is None) or (cond(phi, 1.0) or cond(alt, 1.0)):
            alt.pop(0)
            phi.pop(0)

            n = self.N(phi[0])

            alt_t = p / np.cos(phi[0]) - n
            phi_t = np.arctan(pos[2] / p * (n + alt[0]) / ((n / (1.0 + self.e_snd_sq) + alt[0])))

            alt.append(alt_t)
            phi.append(phi_t)

        return np.array([lam, phi[-1], alt[-1]])

    # pos: [lon, lat, alt] in radians
    def ellipsoidal_to_utm(self, pos):
        if self.gcg is None: self.gcg = rasterio.open(GCG2016_PATH)

        [lon, lat, alt] = pos

        L_rad = np.deg2rad(lon)
        B_rad = np.deg2rad(lat)

        M = 0.9996
        G_0 = 111132.952547
        G_2 = -16038.5088
        G_4 = 16.8326
        G_6 = -0.0220
        L_0 = self.Z * 6 - 183 
        L_d = lon - L_0
        E_0 = ((L_0 + 3.0) / 6.0 + 30.5) * (10**6)
        RHO = 180.0 / np.pi

        t = np.tan(B_rad)
        c = (self.a * self.a) / self.b

        nu_sq = self.e_snd_sq * (np.cos(B_rad)**2)
        N_bar = c / np.sqrt(1.0 + nu_sq) 

        B_cos = np.cos(B_rad)

        corr_1 = M / ((RHO)           ) * N_bar * (B_cos   )
        corr_3 = M / ((RHO**3) * 6.0  ) * N_bar * (B_cos**3) * (1.0 - (t**2) + nu_sq)
        corr_5 = M / ((RHO**5) * 120.0) * N_bar * (B_cos**5) * (5.0 - 18.0 * (t**2) + (t**4) + nu_sq * (14.0 - 58.0 * (t**2)))

        corr_2 = M / ((RHO**2) * 2.0  ) * N_bar * (B_cos**2) * t
        corr_4 = M / ((RHO**4) * 24.0 ) * N_bar * (B_cos**4) * t * (5.0 - (t**2) + 9.0 * nu_sq)
        corr_6 = M / ((RHO**6) * 720.0) * N_bar * (B_cos**6) * t * (61.0 - 58.0 * (t**2) + (t**4))

        G = G_0 * lat + G_2 * np.sin(2.0 * B_rad) + G_4 * np.sin(4.0 * B_rad) + G_6 * np.sin(6.0 * B_rad)

        E = E_0 + corr_1 * L_d + corr_3 * (L_d**3) + corr_5 * (L_d**5)
        N = M * G + corr_2 * (L_d**2) + corr_4 * (L_d**4) + corr_6 * (L_d**6)

        y, x = rasterio.transform.rowcol(self.gcg.transform, lon, lat)
            
        # realistically, all of these values will be the same
        # but I think its better to sample for each point
        # this means if input data had a larger range, it would still be accurate
        H_d = self.gcg.read(1)[y, x]
        H = alt - H_d

        E -= self.Z * 1_000_000
        assert E > 0.0 # if this fails then something is super wrong

        return np.array([E, N, H])

ETRS89_DE = Ellipsoid(6378137.0, np.reciprocal(298.2572221008827112431628366), Z = 32)
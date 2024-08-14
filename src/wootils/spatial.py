import numpy as np
import pyproj as pp



def calc_geodistance(longitude, latitude, ellps='WGS84', **ppkwargs):
    p = pp.Geod(ellps=ellps, **ppkwargs)
    dist = np.full(len(longitude), np.nan)
    dist[1:] = p.inv(longitude[:-1], latitude[:-1], longitude[1:], latitude[1:])[2]
    return dist



def calc_east_north(lon, lat, proj='utm', zone=51, ellps='WGS84', south=True,
                    inverse=False, errcheck=False, radians=False, *ppkwargs):
    p = pp.Proj(proj=proj, zone=zone, ellps=ellps, south=south, *ppkwargs)
    return p(lon, lat, inverse=inverse, errcheck=errcheck, radians=radians)
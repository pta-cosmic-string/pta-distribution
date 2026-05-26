import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Agg") 

# Разрешение
nside = 8  # маленькое для примера, 8 → 768 пикселей
npix = hp.nside2npix(nside)
print("Количество пикселей:", npix)

# Получим координаты центров пикселей
theta, phi = hp.pix2ang(nside, np.arange(npix))
dOmega = 4 * np.pi / npix
# theta — полярный угол [0, π], phi — азимут [0, 2π]

# Преобразуем в единичные векторы
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
directions = np.stack([x, y, z], axis=1)
print("Пример первых 5 направлений:\n", directions[:5])

data = np.arange(npix)
hp.write_map("map.fits", data, overwrite=True)
fits_file = "map.fits"
data = hp.read_map(fits_file)
hp.mollview(data, title="HEALPix сетка", unit="Amplitude", cmap="viridis")
hp.graticule()
plt.show()
# plt.savefig("map.png", dpi=150)
# plt.close()



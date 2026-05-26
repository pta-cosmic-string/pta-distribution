import os
import libstempo as T
from urllib.request import urlretrieve

psrmod_source = 'https://raw.githubusercontent.com/nanograv/enterprise/master/tests/data/mdc1'

def load_psrmod(psrmod='J0030+0451', filepath='data'):
    urlretrieve(f"{psrmod_source}/{psrmod}.par", f"{filepath}/{psrmod}.par")
    urlretrieve(f"{psrmod_source}/{psrmod}.tim", f"{filepath}/{psrmod}.tim")
    psr = T.tempopulsar(
        parfile=f"{filepath}/{psrmod}.par",
        timfile=f"{filepath}/{psrmod}.tim"
    )
    return psr

def open_psrmod(psrmod='J0030+0451', filepath='data'):
    psr = T.tempopulsar(
        parfile=f"{filepath}/{psrmod}.par",
        timfile=f"{filepath}/{psrmod}.tim"
    )
    return psr

def get_psrmod(psrmod='J0030+0451', filepath='data'):
    if not os.path.exists(f"{filepath}/{psrmod}.par") or not os.path.exists(f"{filepath}/{psrmod}.tim"):
        return load_psrmod(psrmod, filepath)
    else:
        return open_psrmod(psrmod, filepath)
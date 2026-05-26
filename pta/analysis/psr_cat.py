import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psrqpy

def load_psrcat(filepath='data'):
    query = psrqpy.QueryATNF()
    psrcat_df = query.pandas
    psrcat_df.to_pickle(f"{filepath}/psrcat.pkl")
    return psrcat_df

def open_psrcat(filepath='data'):
    psrcat_df = pd.read_pickle(f"{filepath}/psrcat.pkl")
    return psrcat_df

def get_psrcat(filepath='data'):
    if not os.path.exists(f"{filepath}/psrcat.pkl"):
        return  load_psrcat(filepath)
    else:
        return open_psrcat(filepath)

def filter_psrcat(psrcat_df, params=['PSRJ', 'RAJD', 'DECJD'], dropnan=True):
    if dropnan:
        psrcat = psrcat_df[params].dropna().to_numpy().T
    else:
        psrcat = psrcat_df[params].to_numpy().T
    return psrcat

def draw_psrcat(rajd_cat, decjd_cat):
    rajd_cat = np.where(rajd_cat>=180, rajd_cat-360, rajd_cat)
    rajd_cat *= np.pi/180
    decjd_cat *= np.pi/180
    plt.figure()
    plt.subplot(projection="mollweide")
    plt.title("Pulsar Catalogue")
    plt.grid(True)
    plt.plot(rajd_cat, decjd_cat, 'o')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    plt.show()

if __name__ == '__main__':
    datadir = 'data'
    psrcat_df = get_psrcat(filepath=datadir)
    psrcat_cat = filter_psrcat(psrcat_df)
    psrj_cat, rajd_cat, decjd_cat = psrcat_cat
    draw_psrcat(rajd_cat, decjd_cat)
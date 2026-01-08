import numpy as np
from scipy.ndimage import sobel, gaussian_filter
#-------------------------------------------------
# SNR reale
#-------------------------------------------------

def snr_real(img, patch_size=32):
 
    H, W = img.shape
    patches = []
# estrae patch in modo  sequenziale su tutta l'immagine e valuta srn localmente
    for y in range(0, H - patch_size, patch_size):
        for x in range(0, W - patch_size, patch_size):
            patches.append(img[y:y+patch_size, x:x+patch_size])

    patches = np.array(patches)
    if len(patches) == 0:      # controlla se non ci sono patch
        return 0.0

    ''' Noise: valuta varianza per ogni patch e prende il 20% più basso  
    var bass - quasi solo rumore
    var alta - dettagli

  '''
    variances = np.var(patches.reshape(len(patches), -1), axis=1)
    noise_thresh = np.percentile(variances, 20)           # percentile = valore che divide la distribuzione
    noise_patches = patches[variances <= noise_thresh]    # seleziono immagini a bassa var 

    if len(noise_patches) == 0:         # vedo se esistono patch di noise , altrimenti 0 
        return 0.0

    noise_std = np.std(noise_patches)

    # gradiente: gx gy e bordi 
  ''' uno mi calcola il gradiente orizzontale l'altro verticale 
  se ci sono bordi verticali, dendriti, cellule-> gx è alto
  se la patch è uniforme -> gx circa 0
  la somma dei quadrati  mi fornisce la media sel segnale

  Analysis of focus measure operators in shape‑from‑focus (Said Pertuz, Puig, García, Pattern Recognition, 2013)
  '''
    grad_energies = []
    for p in patches:
        gx = sobel(p, axis=1)
        gy = sobel(p, axis=0)
        grad_energies.append(np.mean(gx**2 + gy**2))  #energia del bordo e media , per valori alti segnale biologixo      

    grad_energies = np.array(grad_energies)
    signal_thresh = np.percentile(grad_energies, 70)      # valuto percentile, al di sopra del range sono più segnale
    signal_patches = patches[grad_energies >= signal_thresh]  # considera solo quelle al di sorpa della soglia 

    if len(signal_patches) == 0:         # controllo per immagine vuota
        return 0.0

    signal_mean = np.mean(signal_patches)   # livello medio del segnale : la media di intensità nelle patch ad alta struttura

    if noise_std == 0:
        return 0.0

    snr = signal_mean / noise_std                # formula reale
    return float(20 * np.log10(snr + 1e-9))      # conversione in bd

#-------------------------------------------------
# CNR reale
#-------------------------------------------------
   '''
 Patch piatte = rumore
gradienti molto bassi
niente struttura biologica
  Patch strutturate = segnale
gradienti alti
   '''

import numpy as np
from scipy.ndimage import sobel

def cnr_patch_based(img, patch_size=32):
   
    H, W = img.shape     # altezza e larghezza
    patches = []
    gx_list = []
    gy_list = []

    # estrazione patch
    for y in range(0, H - patch_size, patch_size):
        for x in range(0, W - patch_size, patch_size):
            p = img[y:y+patch_size, x:x+patch_size]
            patches.append(p)

            
            gx = sobel(p, axis=1)    # quanto i pixel cambiano da dx a sx 
            gy = sobel(p, axis=0)
            gx_list.append(gx)        # calcolo gradienti per identificare struttura
            gy_list.append(gy)

    patches = np.array(patches)
    gx_list = np.array(gx_list)
    gy_list = np.array(gy_list)

    if len(patches) == 0:
        return 0.0

    # 2) Energia dei gradienti di ogni patch
    grad_energy = np.mean(gx_list**2 + gy_list**2, axis=(1,2))

    # 3) Patch di rumore = 20% con gradiente più basso
    noise_thresh = np.percentile(grad_energy, 20)
    noise_patches = patches[grad_energy <= noise_thresh]

    if len(noise_patches) == 0:
        return 0.0

    # Rumore = deviazione standard del background
    noise_std = np.std(noise_patches)

    # 4) Patch di segnale = 30% con gradiente più alto
    signal_thresh = np.percentile(grad_energy, 70)
    signal_patches = patches[grad_energy >= signal_thresh]

    if len(signal_patches) == 0:
        return 0.0

    # Segnale = media del valore medio nelle patch strutturate
    signal_mean = np.mean(signal_patches)

    # 5) Background = media del valore medio delle patch piatte
    background_mean = np.mean(noise_patches)

    # CNR = contrasto / rumore
    cnr = (signal_mean - background_mean) / (noise_std + 1e-8)

    return float(cnr)
"""
 binnig = finetra su FOV per triggereare monivemto tr 2 frame successivi , centri il quadrante con piu' 


"""

import numpy as np
from scipy.ndimage import sobel, gaussian_filter
'''
misura l'anisotropia basata sul tensore di struttura
ottengo un valore  in [0, 1], dove:
        1   = struttura altamente orientata (fibre, neuriti, bordi netti) λ₁ ≫ λ₂
        0.5 = struttura moderatamente orientata λ₁ > λ₂
        0   = isotropia totale (rumore, texture casuale) λ₁ ≈ λ₂
'''



def structure_tensor_coherence(img, sigma=1.5):  #immagine 2d normalizzata in [0,1] , filtro gaus
   

    # gradiente orizzontale e verticale
    gx = sobel(img, axis=1)   # variazioni lungo x  
    gy = sobel(img, axis=0)   # variazioni lungo y  

    # componenti del tensore di struttura
    Jxx = gaussian_filter(gx * gx, sigma)
    Jyy = gaussian_filter(gy * gy, sigma)
    Jxy = gaussian_filter(gx * gy, sigma)

    # autovalori λ1 ≥ λ2 del tensore : quanta info esiste sulla direz principale e quanta ne rimane sulla ortigonale 
  # se la differenza è grande la struttura è orientata altrimenti - isotropia - rumore
    # Formula esplicita 
    tmp = np.sqrt((Jxx - Jyy)**2 + 4*(Jxy**2))
    lam1 = ((Jxx + Jyy) + tmp) / 2.0
    lam2 = ((Jxx + Jyy) - tmp) / 2.0
'''
λ 1 Direzione principale della struttura locale (es: orientamento di un neurite, fibra, bordo)
λ 2 (piccolo) Direzione ortogonale, contiene poca informazione (es: direzione in cui la struttura non varia)
'''
    #coerenza: misura di anisotropia produce val tra 0 e 1  se =1 massima anisotropia 
    #    (lambda1 - lambda2) / (lambda1 + lambda2)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-12)

    # coerenza media dell’immagine
    return float(np.mean(coherence))

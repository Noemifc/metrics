'''
Ideata per rilevare un danno brusco ovvero: l’immagine cambia molto da un frame all'altro,
significa che è accaduto qualcosa di fisico al campione
SSIM confronta: luminanza , contrasto e struttura
'''

'''
per SSIM che varia in [0, 1]
Score proposti tra i frame successivi 
Valore	Significato
1.0	identici
0.95–0.8	variazione lieve
0.8–0.6	cambiamento significativo
< 0.6	danno brusco
< 0.4	collasso evidente / rottura forte
'''


def similarity_ssim(img_a: np.ndarray, img_b: np.ndarray, eps: float = 1e-6) -> float:

    if img_a is None or img_b is None:
        return 1.0  # identiche
    
    if img_a.shape != img_b.shape:
        return 1.0  # per non segnalare falsi pos dovuti alla forma 

    # Conversione float32  
    A = img_a.astype(np.float32)
    B = img_b.astype(np.float32)

    # media
    muA = A.mean()
    muB = B.mean()

    # varianza
    sigmaA = A.var()
    sigmaB = B.var()

    # Cov
    sigmaAB = np.mean((A - muA) * (B - muB))
'''
paper: 10.1109/TIP.2003.819861
Se la varianza è bassa -> img uniformr -> inserisco C1 e C2 costanti di stabilizzazione che impediscono div!0 o ampl rumore o instabilità numeriche
C = (K * L)^2: 
L 0 range dinamico per immagin norm L=1 e K1 = 0.01  K2 = 0.03 come da paper 
'''
    # Costanti di stabilizzazione (da SSIM paper) e robusto 
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
# luminanza : quanto le due immagini hanno la stessa luminanza media
    numerator = (2 * muA * muB + C1) * (2 * sigmaAB + C2)

# normalizzazione luminanza
    denominator = (muA**2 + muB**2 + C1) * (sigmaA + sigmaB + C2)

#Ssim= somiglianza tra immagini / differenze nel confronto 
    ssim = numerator / (denominator + eps)
    return float(max(0.0, min(1.0, ssim)))  # forza il valore finale a cadere sempre nell’intervallo [0, 1]

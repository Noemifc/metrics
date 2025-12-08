'''
Usando SSIM da base  vede quanto il campione si è degradato rispetto al PRIMO frame, è una metrica cumulativa
mi indica quanto il campione è peggiorato dall'inizio
Richidede primo frame stabile
   Damage = 1 - SSIM(img_curr, img_first)

    0.0   nessun danno
    0.2   degradazione lieve
    0.4   degrado moderato
    0.6+  forte danno strutturale
'''
# Funzione metrica da aggiungere alle altre

def radiation_damage_score(img_first: np.ndarray,
                           img_curr: np.ndarray,
                           eps: float = 1e-6) -> float:
   
    if img_first is None or img_curr is None:
        return 0.0

    if img_first.shape != img_curr.shape:
        return 0.0

    A = img_first.astype(np.float32)
    B = img_curr.astype(np.float32)

    muA = A.mean()
    muB = B.mean()

    sigmaA = A.var()
    sigmaB = B.var()

    sigmaAB = np.mean((A - muA) * (B - muB))

    # Stabilizing constants (from original SSIM paper)
    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2 * muA * muB + C1) * (2 * sigmaAB + C2)
    denominator = (muA**2 + muB**2 + C1) * (sigmaA + sigmaB + C2)

    ssim = numerator / (denominator + eps)
    ssim = max(0.0, min(1.0, ssim))

    return float(1.0 - ssim)

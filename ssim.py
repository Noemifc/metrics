import numpy as np
import os

OUT_PATH = "/home/beams/NFICO/Desktop/ssim.txt"  # file di testo esterno

def similarity_ssim(img_a: np.ndarray, img_b: np.ndarray, eps: float = 1e-6) -> float:
    if img_a is None or img_b is None:
        return 1.0
    if img_a.shape != img_b.shape:
        return 1.0

    A = img_a.astype(np.float32)
    B = img_b.astype(np.float32)

    muA = A.mean()
    muB = B.mean()

    sigmaA = A.var()
    sigmaB = B.var()

    sigmaAB = np.mean((A - muA) * (B - muB))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * muA * muB + C1) * (2 * sigmaAB + C2)
    denominator = (muA**2 + muB**2 + C1) * (sigmaA + sigmaB + C2)

    ssim = numerator / (denominator + eps)
    return float(max(0.0, min(1.0, ssim)))


def make_process(out_path: str = OUT_PATH):
    buffered_frames = []

    # crea la cartella se non esiste
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # apre il file una volta sola (più efficiente)
    f = open(out_path, "a", buffering=1)  # buffering=1 -> line buffered

    def process(img: np.ndarray) -> np.ndarray:
        nonlocal buffered_frames
        buffered_frames.append(img)

        if len(buffered_frames) == 1:
            return img

        score = similarity_ssim(buffered_frames[-2], buffered_frames[-1])
        buffered_frames = buffered_frames[-2:]

        print(score)
        f.write(f"{score}\n")  # salva nel file

        return img

    def reset():
        nonlocal buffered_frames
        buffered_frames = []
        f.flush()

    return process, reset


process, reset = make_process()




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

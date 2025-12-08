

''' 
misura di nitidezza basata sui gradienti
'''

def tenengrad_focus(img):      #restituisce un valore numerico che rappresenta la nitidezza
 
    gx = sobel(img, axis=1)   # applica filtro sobel e individua i bordi 
    gy = sobel(img, axis=0)
    G = gx**2 + gy**2         # energia del gradiente : amplifica la differenza tra immagini nitide e sfocate
    return float(np.mean(G))  # media ai bordi: valore alto -> immagine a fuoco 

'''
brenner valuta la nitidezza dell’immagine controllando quanto cambiano i pixel lungo una direzione
'''
def brenner_focus(img):

    return float(np.mean((img[:, 2:] - img[:, :-2])**2)) # Diff a distanza di 2 pixel - quadrato - media - nitidezza

'''
valuta la densità dei bordi 
'''
def edge_density(img, threshold=0.1):

    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    edge_pixels = np.sum(mag > threshold)
    return float(edge_pixels / img.size)


def lbp_uniformity(img):
    """Local Binary Pattern uniformity score."""
    H, W = img.shape
    u_count = 0
    transitions = []

    for y in range(1, H-1):
        for x in range(1, W-1):
            c = img[y, x]
            nb = [
                img[y-1, x-1], img[y-1, x], img[y-1, x+1],
                img[y, x+1], img[y+1, x+1], img[y+1, x],
                img[y+1, x-1], img[y, x-1]
            ]
            binary = [1 if n > c else 0 for n in nb]

            # Count transitions in circular binary pattern
            t = sum(binary[i] != binary[(i+1) % 8] for i in range(8))
            transitions.append(t)

    # Uniform LBP = transitions <= 2
    uniform = np.sum(np.array(transitions) <= 2)
    return float(uniform / len(transitions))


def structure_tensor_coherence(img, sigma=1.5):
    """Anisotropy / coherence from structure tensor."""
    gx = sobel(img, axis=1)
    gy = sob

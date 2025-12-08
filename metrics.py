

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
valuta la densità dei bordi e quanta morfologia vi è nell'immagine 
vicino a 0.0  immagine piatta, pochi bordi
vicino a 1.0  immagine ricca di struttura
'''
def edge_density(img, threshold=0.1):
    gx = sobel(img, axis=1)             # bordi verticali e poi orizzontali
    gy = sobel(img, axis=0)
    mag = np.sqrt(gx**2 + gy**2)        #forza del bordo: alto se tra pixel adiacenti c’è una grande diff di intensità (bordo forte)
    edge_pixels = np.sum(mag > threshold) 
    return float(edge_pixels / img.size) # rapporto tra bordi e tot

''' quanto l'immagine è localmente uniforme texture analisys
'''
def lbp_uniformity(img):

    H, W = img.shape
    u_count = 0
    transitions = []   # salvaper ogni pixel, il numero di transizioni 0-1 e 1-0 nella lista transition

    for y in range(1, H-1):      # vede pixel escludendo bordi 
        for x in range(1, W-1):
            c = img[y, x]        # pixel centrale
            nb = [ 
                img[y-1, x-1], img[y-1, x], img[y-1, x+1],
                img[y, x+1], img[y+1, x+1], img[y+1, x],
                img[y+1, x-1], img[y, x-1]
            ]                                               # ordine antiorario
            binary = [1 if n > c else 0 for n in nb]        # vicino confrontato con il pixel centrale  - ho pattern  8 bit
                                                            # se il vicino è più luminoso del centro - 1
        ''' t= conta quante volte il bit cambia valore rispetto al successivo'''  
            t = sum(binary[i] != binary[(i+1) % 8] for i in range(8))    # numero di transizioni attorno al pixel + confronto 'ultimo bit con il primo 
            transitions.append(t)                                        # (i+1) % 8 = confronto ultimo bit con primo 

    # per  LBP uniforme  = transitions <= 2
    uniform = np.sum(np.array(transitions) <= 2)  # numero di pixel in cui t <= 2
    return float(uniform / len(transitions))      #pixel analizzati 



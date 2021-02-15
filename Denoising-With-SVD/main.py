#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cut_mat(s, n):
    S = np.zeros((s.shape[0], s.shape[1]))
    for i in range(n):
        S[i] = s[i]
    return S
img = Image.open("noisy.jpg")
noisy_matrix = np.asarray(img)
U_r, s_r, V_r = np.linalg.svd(noisy_matrix[:,:,0], full_matrices=False)
U_g, s_g, V_g = np.linalg.svd(noisy_matrix[:,:,1], full_matrices=False)
U_b, s_b, V_b = np.linalg.svd(noisy_matrix[:,:,2], full_matrices=False)
"""
find best


for i in range(3, 21):
    img_r = np.matrix(U_r) * cut_mat(np.diag(s_r), i) * np.matrix(V_r)
    img_g = np.matrix(U_g) * cut_mat(np.diag(s_g), i) * np.matrix(V_g)
    img_b = np.matrix(U_b) * cut_mat(np.diag(s_b), i) * np.matrix(V_b)
    img = np.zeros((img_r.shape[0],img_r.shape[1],3))
    img [:,:,0] = img_r
    img [:,:,1] = img_g
    img [:,:,2] = img_b
    img_to_show = img/255
    plt.imshow(img_to_show)
    plt.title(str(n))
    plt.show()
"""

i = 18
img_r = np.matrix(U_r) * cut_mat(np.diag(s_r), i) * np.matrix(V_r)
img_g = np.matrix(U_g) * cut_mat(np.diag(s_g), i) * np.matrix(V_g)
img_b = np.matrix(U_b) * cut_mat(np.diag(s_b), i) * np.matrix(V_b)
img = np.zeros((img_r.shape[0],img_r.shape[1],3))
img [:,:,0] = img_r
img [:,:,1] = img_g
img [:,:,2] = img_b
img = img.astype(np.uint8)
plt.imsave('denoised.jpeg', img)

### /\ /V\ [] [Z [Z (≡ ¯/_ /\ [Z /\ _] /\ !3 [] ###

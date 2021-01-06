#Import Libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Get inputs
def get():
    flag = 0
    pic_arr = 0
    name = input("Image path:")
    landa = float(input("lambda (λ):"))
    try:
        pic = Image.open(name)
        pic_arr = np.asarray(pic)
        flag = 1
    except:
        print("there is no file :|")
    return pic_arr, landa, flag

#show picture
def show(pic):
    plt.imshow(pic)
    plt.show()

#make shadow with
def make_shadow_array(pic_arr, landa):
    shadow = np.zeros((pic_arr.shape[0], pic_arr.shape[1] + int(landa * pic_arr.shape[0]), 3)) + 1
    for i in range(pic_arr.shape[0]):
        for j in range(pic_arr.shape[1]):
            """
            for better result can use it
            if(not(pic_arr[i,j,0] > 245 and pic_arr[i,j,1] > 245 and pic_arr[i,j,2] > 245)):
            """
            if(not(pic_arr[i,j,0] == 255 and pic_arr[i,j,1] == 255 and pic_arr[i,j,2] == 255)):
                shadow[i,j+int(landa*i)] = [0.3, 0.3,0.3]
    return shadow

#fix shadow
def fix_shadow(shadow, pic_arr):
    for i in range(pic_arr.shape[0]):
        for j in range(pic_arr.shape[1]):
            """
            for better result can use it
            if(not(pic_arr[i,j,0] > 245 and pic_arr[i,j,1] > 245 and pic_arr[i,j,2] > 245)):
            """
            if(not(pic_arr[i,j,0] == 255 and pic_arr[i,j,1] == 255 and pic_arr[i,j,2] == 255)):
                shadow[i,j] = pic_arr[i,j]/255
    

#main
Get = get()
if(Get[2] == 1):
    shadow = make_shadow_array(Get[0], Get[1])
    fix_shadow(shadow, Get[0])
    show(shadow)

# /\ /V\ [] [Z [Z (≡ ¯/_ /\ [Z /\ _] /\ !3 [] #

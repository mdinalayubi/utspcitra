import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Mendefinisikan fungsi detektor
def Canny_detector(img, weak_th=None, strong_th=None):
    # Konversi gambar ke skala abu-abu
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Langkah pengurangan kebisingan
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Menghitung gradien
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize=3)
    
    # Konversi koordinat kartesius ke kutub
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Mengatur ambang batas minimum dan maksimum untuk ambang batas ganda
    mag_max = np.max(mag)
    if weak_th is None:
        weak_th = mag_max * 0.1
    if strong_th is None:
        strong_th = mag_max * 0.5
    
    # Mendapatkan dimensi gambar masukan
    height, width = img.shape
    
    # Mengulangi setiap piksel skala abu-abu gambar
    for i_x in range(width):
        for i_y in range(height):
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)
            
            # Memilih tetangga piksel target sesuai dengan arah gradien
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x+1, i_y
            elif grad_ang > 22.5 and grad_ang <= 67.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x+1, i_y+1
            elif grad_ang > 67.5 and grad_ang <= 112.5:
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y+1
            elif grad_ang > 112.5 and grad_ang <= 157.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y+1
                neighb_2_x, neighb_2_y = i_x+1, i_y-1
            
            # Langkah penekanan tidak maksimal
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue
            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0
    
    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)
    
    # Langkah ambang batas ganda
    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2
    
    # Mengembalikan besarnya gradien tepi
    return mag

# Membaca gambar
frame = cv2.imread('food.jpeg')

# Memanggil fungsi yang dirancang untuk menemukan tepian
canny_img = Canny_detector(frame)

# Menampilkan gambar masukan dan keluaran
plt.figure()
f, plots = plt.subplots(2, 1, figsize=(10, 10))
plots[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plots[0].set_title('Input Image')
plots[1].imshow(canny_img, cmap='gray')
plots[1].set_title('Canny Edge Detection')
plt.show()
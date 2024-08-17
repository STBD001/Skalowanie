import numpy as np
import cv2

# Wczytanie obrazu i konwersja do przestrzeni kolorów RGB
img1 = cv2.cvtColor(cv2.imread('Julka.bmp'), cv2.COLOR_BGR2RGB)
scale_factor = 0.5

# Wyznaczenie wymiarów oryginalnego obrazu
height, width, channels = img1.shape

# Wyznaczenie nowych wymiarów dla zmniejszonego obrazu (przez 2)
new_height, new_width = height // 2, width // 2

# Wyznaczenie nowych wymiarów dla powiększonego obrazu (również przez 2)
new2_height, new2_width = height * 2, width * 2

# Inicjalizacja macierzy dla powiększonego obrazu
enlarged_img = np.zeros((new2_height, new2_width, channels), dtype=np.uint8)

# Iteracja po nowym obrazie (powiększonym)
for w in range(new2_height):
    for k in range(new2_width):
        for c in range(channels):
            # Współrzędne pikseli w oryginalnym obrazie
            x = k // 2
            y = w // 2

            # Interpolacja pikseli w oryginalnym obrazie dla nowego piksela
            enlarged_img[w, k, c] = img1[y, x, c]

# Inicjalizacja macierzy dla zmniejszonego obrazu
reduc_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

# Iteracja po nowym obrazie (zmniejszonym)
for w in range(new_height):
    for k in range(new_width):
        for c in range(channels):
            x = k * 2
            y = w * 2
            # Interpolacja pikseli w oryginalnym obrazie dla nowego piksela (uśrednianie)
            reduc_img[w, k, c] = (
                    (img1[y, x, c]) // 4 +
                    (img1[y, x + 1, c]) // 4 +
                    (img1[y + 1, x, c]) // 4 +
                    (img1[y + 1, x + 1, c]) // 4
            )

# Obracanie obrazu o 45 stopni
rows, cols, _ = img1.shape
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(img1, rotation_matrix, (cols, rows), flags=cv2.INTER_LINEAR)

# Zmiana rozmiaru obrazu po obróceniu
resized_rotated_image = cv2.resize(rotated_image, (512, 512), interpolation=cv2.INTER_LINEAR)

# Wyświetlenie oryginalnego, obróconego, zmniejszonego i powiększonego obrazu
cv2.imshow('Original', img1)
cv2.imshow('Turned', resized_rotated_image)
cv2.imshow('Smaller', reduc_img)
cv2.imshow('Bigger', enlarged_img)

# Oczekiwanie na naciśnięcie klawisza i zamknięcie okien
cv2.waitKey(0)
cv2.destroyAllWindows()
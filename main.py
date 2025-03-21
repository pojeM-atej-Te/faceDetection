import cv2 as cv
import numpy as np


def zmanjsaj_sliko(slika, sirina, visina):
    '''Zmanjšaj sliko na velikost sirina x visina.'''
    return cv.resize(slika, (sirina, visina))

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:
    '''Sprehodi se skozi sliko v velikosti škatle (sirina_skatle x visina_skatle) in izračunaj število pikslov kože v vsaki škatli.
    Škatle se ne smejo prekrivati!
    Vrne seznam škatel, s številom pikslov kože.
    Primer: Če je v sliki 25 škatel, kjer je v vsaki vrstici 5 škatel, naj bo seznam oblike
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]].
      V tem primeru je v prvi škatli 1 piksel kože, v drugi 0, v tretji 0, v četrti 1 in v peti 1.'''
    pass


def prestej_piklse_z_barvo_koze(slika, barva_koze) -> int:
    '''Prestej število pikslov z barvo kože v škatli.'''
    pass


def doloci_barvo_koze(slika, levo_zgoraj, desno_spodaj):
    """
    Izračuna spodnje in zgornje meje barve kože na izbranem območju slike.

    Args:
        slika: 3D tabela/polje tipa np.ndarray
        levo_zgoraj: tuple (x,y), ki vsebuje skrajno zgornjo levo koordinato oklepajočega kvadrata obraza
        desno_spodaj: tuple (x,y), ki vsebuje skrajno spodnjo desno koordinato oklepajočega kvadrata obraza

    Returns:
        tuple (spodnja_meja, zgornja_meja): Spodnja in zgornja meja barve kože (np.ndarray)
    """
    # Izreži območje obraza iz slike
    x1, y1 = levo_zgoraj
    x2, y2 = desno_spodaj

    # Preveri pravilnost koordinat
    if x1 > x2 or y1 > y2:
        raise ValueError("Nepravilne koordinate: levo_zgoraj mora biti nad in levo od desno_spodaj")

    obraz = slika[y1:y2, x1:x2]

    # Izračunaj povprečje barve kože v BGR prostoru
    mean_color = np.mean(obraz, axis=(0, 1))

    # Določi meje barve kože (povprečje +/- 40%)
    margin = 0.4
    spodnja_meja = np.maximum(mean_color * (1 - margin), 0).astype(np.uint8)
    zgornja_meja = np.minimum(mean_color * (1 + margin), 255).astype(np.uint8)

    return (spodnja_meja, zgornja_meja)

def main():
    # Initialize the camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the size of the region of interest (ROI)
    roi_width = 200
    roi_height = 200

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = zmanjsaj_sliko(frame, 220, 340)
        frame = cv.flip(frame, 90)

        # Get the dimensions of the frame
        frame_height, frame_width = frame.shape[:2]

        # Calculate the center of the frame
        center_x, center_y = frame_width // 2, frame_height // 2

        # Define the ROI start and end points
        roi_start = (center_x - roi_width // 2, center_y - roi_height // 2)
        roi_end = (center_x + roi_width // 2, center_y + roi_height // 2)

        # Draw the rectangle on the frame
        cv.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)

        # Display the resulting frame
        cv.imshow('Camera', frame)

        # Wait for key press
        key = cv.waitKey(1) & 0xFF
        if key == ord('r'):
            # Capture the image within the ROI
            roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
            cv.imshow('Captured Image', roi)
        elif key == 27:  # Escape key
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    #Pripravi kamero

    # Zajami prvo sliko iz kamere

    # Izračunamo barvo kože na prvi sliki

    # Zajemaj slike iz kamere in jih obdeluj

    # Označi območja (škatle), kjer se nahaja obraz (kako je prepuščeno vaši domišljiji)
    # Vprašanje 1: Kako iz števila pikslov iz vsake škatle določiti celotno območje obraza (Floodfill)?
    # Vprašanje 2: Kako prešteti število ljudi?

    # Kako velikost prebirne škatle vpliva na hitrost algoritma in točnost detekcije? Poigrajte se s parametroma velikost_skatle
    # in ne pozabite, da ni nujno da je škatla kvadratna.
    main()
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
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]].'''

    # Initialize the list to store the number of skin pixels in each box
    skatle = []

    # Get the dimensions of the image
    visina_slike, sirina_slike = slika.shape[:2]

    # Loop through the image in steps of the box size
    for y in range(0, visina_slike, visina_skatle):
        vrstica = []
        for x in range(0, sirina_slike, sirina_skatle):
            # Define the current box
            skatla = slika[y:y + visina_skatle, x:x + sirina_skatle]

            # Count the number of skin color pixels in the current box
            st_pikslov_koze = prestej_piklse_z_barvo_koze(skatla, barva_koze)
            vrstica.append(st_pikslov_koze)

        # Append the row to the list of boxes
        skatle.append(vrstica)

    return skatle

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

            # Calculate the skin color in the ROI
            spodnja_meja, zgornja_meja = doloci_barvo_koze(frame, roi_start, roi_end)
            print(f"Spodnja meja barve kože: {spodnja_meja}")
            print(f"Zgornja meja barve kože: {zgornja_meja}")

            # Overlay the captured ROI on the original frame
            frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = roi

            # Display the captured image
            cv.imshow('Captured Image', frame)
            cv.waitKey(0)  # Wait indefinitely until a key is pressed
            break  # Exit the loop after capturing the image

        if key == 27:  # Escape key
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
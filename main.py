import cv2 as cv
import numpy as np
import time

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
            st_pikslov_koze = prestej_piksle_z_barvo_koze(skatla, barva_koze)
            vrstica.append(st_pikslov_koze)

        # Append the row to the list of boxes
        skatle.append(vrstica)

    return skatle

def prestej_piksle_z_barvo_koze(skatla, barva_koze) -> int:
    '''Preštej število pikslov z barvo kože v škatli.'''
    spodnja_meja, zgornja_meja = barva_koze

    # Ustvari masko z vrednostmi 1 za piksle v intervalu barve kože
    maska = cv.inRange(skatla, spodnja_meja, zgornja_meja)

    # Uporabi morfološke operacije za odstranjevanje šuma
    kernel = np.ones((5, 5), np.uint8)
    maska = cv.morphologyEx(maska, cv.MORPH_OPEN, kernel)
    maska = cv.morphologyEx(maska, cv.MORPH_CLOSE, kernel)

    # Preštej število pikslov
    return cv.countNonZero(maska)


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
    spodnja_meja = np.maximum(mean_color * (1 - margin), 0).astype(np.uint8).reshape(1, 3)
    zgornja_meja = np.minimum(mean_color * (1 + margin), 255).astype(np.uint8).reshape(1, 3)

    return (spodnja_meja, zgornja_meja)

def main():
    # Inicializacija kamere
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Napaka: Kamera ni na voljo.")
        return

    # Določitev velikosti območja interesa (ROI)
    roi_width = 200
    roi_height = 200
    barva_koze = None
    barva_koze_inicializirana = False

    while True:
        # Zajem slike
        ret, frame = cap.read()
        if not ret:
            print("Napaka: Ne morem prebrati okvirja.")
            break

        frame = zmanjsaj_sliko(frame, 220, 340)
        frame = cv.flip(frame, 90)
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        roi_start = (center_x - roi_width // 2, center_y - roi_height // 2)
        roi_end = (center_x + roi_width // 2, center_y + roi_height // 2)

        if not barva_koze_inicializirana:
            # Izriši pravokotnik za določitev barve kože
            cv.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)
            cv.putText(frame, "Poravnajte obraz in pritisnite 'r'", (10, frame_height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # Obdelava slike s škatlami
            start_time = time.time()
            skatle = obdelaj_sliko_s_skatlami(frame, 50, 50, barva_koze)
            for y, vrstica in enumerate(skatle):
                for x, st_pikslov in enumerate(vrstica):
                    if st_pikslov > 0:
                        cv.rectangle(frame, (x * 50, y * 50), ((x + 1) * 50, (y + 1) * 50), (0, 255, 0), 2)
            processing_time = time.time() - start_time
            cv.putText(frame, f"Cas obdelave: {processing_time * 1000:.1f} ms", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv.imshow('Kamera', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('r'):
            # Zajem slike znotraj ROI
            roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
            spodnja_meja, zgornja_meja = doloci_barvo_koze(frame, roi_start, roi_end)
            barva_koze = (spodnja_meja, zgornja_meja)
            barva_koze_inicializirana = True
            print(f"Spodnja meja barve kože: {spodnja_meja}")
            print(f"Zgornja meja barve kože: {zgornja_meja}")
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
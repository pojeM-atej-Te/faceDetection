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

    #Inicializacija seznama škatel
    skatle = []

    #vrne obliko slike v obliki (visina, sirina, kanali), kjer kanali predstavljajo barvne komponente (BGR).
    visina_slike, sirina_slike = slika.shape[:2]

    prag_pikslov = 80
    #loopa skozi sliko
    for y in range(0, visina_slike, visina_skatle): #dobimo vrstice
        vrstica = []
        for x in range(0, sirina_slike, sirina_skatle):#dobimo stolpce

            #Definira škatlo
            skatla = slika[y:y + visina_skatle, x:x + sirina_skatle]

            #Preštej število pikslov kože v škatli
            st_pikslov_koze = prestej_piksle_z_barvo_koze(skatla, barva_koze)
            if st_pikslov_koze >= prag_pikslov:
                #če je število pikslov večje od praga dodaj, sicer ne
                vrstica.append(1)
            else:
                vrstica.append(0)

        # Dodaj vrstico v seznam škatel
        skatle.append(vrstica)
    return skatle

def prestej_piksle_z_barvo_koze(skatla, barva_koze) -> int:
    '''Preštej število pikslov z barvo kože v škatli.'''
    spodnja_meja, zgornja_meja = barva_koze

    # Ustvari masko z vrednostmi 255 ce je piksel znotraj intervala, 0 ce ni
    maska = cv.inRange(skatla, spodnja_meja, zgornja_meja)

    # Preštej število belih pikslov v maski
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
    x1, y1 = levo_zgoraj
    x2, y2 = desno_spodaj

    # Preveri pravilnost koordinat, x1 mora biti manjši od x2 in y1 manjši od y2, x1 je levo od x2 in y1 nad y2
    if x1 > x2 or y1 > y2:
        raise ValueError("Nepravilne koordinate: levo_zgoraj mora biti nad in levo od desno_spodaj")

    # Izreži Region of Interest (visina x širina x 3 barvni kanali)
    obraz = slika[y1:y2, x1:x2]

    obraz.reshape(-1, 3) #pretvori se v matriko 3 stolpci za RGB za lažje računanje povprečjas

    # Izračunaj povprečje barve kože v BGR prostoru
    mean_color = np.mean(obraz, axis=(0, 1))
    #izpiše povprečje RGB vrednosti npr [ 90.  120.  150.]

    # Določi meje barve kože (povprečje +/- 50%)
    margin = 0.5
    #samo nenegativne vrednosti od 0-255 np.uint8
    spodnja_meja = np.maximum(mean_color * (1 - margin), 0).astype(np.uint8).reshape(1, 3)
    zgornja_meja = np.minimum(mean_color * (1 + margin), 255).astype(np.uint8).reshape(1,3)

    return spodnja_meja, zgornja_meja

def main():
    # Inicializacija kameres
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Napaka: Kamera ni na voljo.")
        return

    # Določitev velikosti območja interesa (ROI)
    roi_width = 80
    roi_height = 150
    barva_koze = None
    barva_koze_inicializirana = False

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        # Zajem trenutne kamere
        ret, frame = cap.read()
        if not ret:
            print("Napaka: Ne morem prebrati okvirja.")
            break

        frame = zmanjsaj_sliko(frame, 240, 320)
        frame = cv.flip(frame, 1) #zrcaljenje slike
        frame_height, frame_width = frame.shape[:2] #dobimo višino in širino slike
        center_x, center_y = frame_width // 2, frame_height // 2 #dobimo sredino slike
        roi_start = (center_x - roi_width // 2, center_y - roi_height // 2)#določimo začetek ROI
        roi_end = (center_x + roi_width // 2, center_y + roi_height // 2)#določimo konec ROI

        # Velikost škatle glede na velikost slike
        sirina_skatle = int(frame_width * 0.10)
        visina_skatle = int(frame_height * 0.10)

        if not barva_koze_inicializirana:
            # Izriši pravokotnik za določitev barve kože
            cv.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)
            cv.putText(frame, "Poravnajte obraz in pritisnite 'r'", (10, frame_height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # Obdelava slike s škatlami
            start_time = time.time()
            skatle = obdelaj_sliko_s_skatlami(frame, sirina_skatle, visina_skatle, barva_koze)
            for y, vrstica in enumerate(skatle):
                for x, st_pikslov in enumerate(vrstica):
                    if st_pikslov > 0:
                        cv.rectangle(frame, (x * sirina_skatle, y * visina_skatle), ((x + 1) * sirina_skatle, (y + 1) * visina_skatle), (0, 255, 0), 2)
            processing_time = time.time() - start_time
            cv.putText(frame, f"Cas obdelave: {processing_time * 1000:.1f} ms", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time > 1: #ali je že minila sekunda
            fps = frame_count / elapsed_time #izračunamo fps
            frame_count = 0 #ponastavimo števec
            prev_time = current_time #posodobimo čas

        #Izpiše fpsje
        cv.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
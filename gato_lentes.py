import cv2
import os

# ==== CONFIGURACIÓN ====

# Rutas clasificadores Haar
haar_human = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
haar_cat = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'haarcascade_frontalcatface.xml'))

# Cargar imágenes de lentes
lentes_dir = os.path.join(os.getcwd(), 'lentes')
lentes_png = sorted([
    os.path.join(lentes_dir, f) for f in os.listdir(lentes_dir)
    if f.endswith('.png')
])
total_lentes = len(lentes_png)
lente_index = 0

# Imagen fija
imagen_fija_path = os.path.join(os.getcwd(), 'original_bobby.jpg')

# ==== FUNCIONES ====
def superponer_png(base_img, overlay_png, x, y, ancho_rostro, alto_rostro):
    """ Superpone una imagen PNG con canal alfa sobre una imagen base,
    manteniendo la proporción original y centrando sobre la zona ocular. """
    overlay = cv2.imread(overlay_png, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.shape[2] != 4:
        return base_img

    # Escalar a 75% del ancho del rostro
    new_w = int(ancho_rostro * 0.75)

    # Calcular altura proporcional según aspecto original
    orig_h, orig_w = overlay.shape[:2]
    aspect_ratio = orig_w / orig_h
    new_h = int(new_w / aspect_ratio)

    overlay = cv2.resize(overlay, (new_w, new_h))

    # Posicionar sobre los ojos: mover hacia arriba 20% del alto del rostro
    x_offset = x + (ancho_rostro - new_w) // 2
    y_offset = y + int(alto_rostro * 0.25)

    # Recortes para evitar desbordes
    if y_offset + new_h > base_img.shape[0] or x_offset + new_w > base_img.shape[1]:
        return base_img

    # Separar canales
    b, g, r, a = cv2.split(overlay)
    mask = cv2.merge((a, a, a))
    fg = cv2.merge((b, g, r))
    roi = base_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]

    try:
        fg = cv2.bitwise_and(fg, mask)
        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
        dst = cv2.add(bg, fg)
        base_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = dst
    except:
        pass

    return base_img

def procesar_imagen(img, lente_path):
    """ Procesa una imagen, detectando rostros y superponiendo gafas. """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detección de rostro humano
    caras = haar_human.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in caras:
        img = superponer_png(img, lente_path, x, y + int(h * 0.25), w, int(h * 0.3))

    # Detección de rostro de gato
    gatos = haar_cat.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in gatos:
        img = superponer_png(img, lente_path, x, y + int(h * 0.3), w, int(h * 0.25))

    return img

# ==== IMAGEN FIJA O WEBCAM ====
def ejecutar_modo_imagen():
    img = cv2.imread(imagen_fija_path)
    if img is None:
        print("No se pudo cargar la imagen fija.")
        return
    img = redimensionar(img)
    procesada = procesar_imagen(img, lentes_png[lente_index])
    cv2.imshow('Imagen Fija con Filtro', procesada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ejecutar_modo_webcam():
    global lente_index
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = redimensionar(frame)
        frame = procesar_imagen(frame, lentes_png[lente_index])
        cv2.imshow('Webcam con Filtro', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):  # Cambiar lentes
            lente_index = (lente_index + 1) % total_lentes

    cap.release()
    cv2.destroyAllWindows()

def redimensionar(img, max_ancho=800):
    """ Redimensiona imagen para que no sea demasiado grande. """
    if img.shape[1] > max_ancho:
        escala = max_ancho / img.shape[1]
        img = cv2.resize(img, (0,0), fx=escala, fy=escala)
    return img

# ==== MENÚ PRINCIPAL EN TERMINAL ====
if __name__ == '__main__':
    print("Presiona 'i' para imagen fija, 'w' para webcam, 'ESC' o 'q' para salir.")
    print("Durante webcam: pulsa 'c' para cambiar lentes.")

    while True:
        key = input("¿Qué deseas hacer? (i=imagen, w=webcam, q=salir): ").lower()
        if key == 'i':
            ejecutar_modo_imagen()
        elif key == 'w':
            ejecutar_modo_webcam()
        elif key == 'q':
            print("Saliendo...")
            break
        else:
            print("Opción inválida.")


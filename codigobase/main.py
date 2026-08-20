import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

ojos_cerrados_desde = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = face_mesh.process(rgb)

    estado = "OK"

    if resultado.multi_face_landmarks:
        # simulación simple
        ojos_cerrados = False

        if ojos_cerrados:
            if ojos_cerrados_desde is None:
                ojos_cerrados_desde = time.time()
            elif time.time() - ojos_cerrados_desde > 3:
                estado = "ALERTA"
        else:
            ojos_cerrados_desde = None

    cv2.putText(
        frame,
        estado,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Monitoreo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

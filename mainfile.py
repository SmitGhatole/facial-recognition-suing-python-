import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

counter = 0
face_match = False
reference_img = cv2.imread("reference.jpg")

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())["verified"]:
            face_match = True
        else:
            face_match = False
    except Exception as e:
        print(f"Error in face verification: {e}")
        face_match = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if counter % 30 == 0:
        try:
            threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()
        except ValueError as e:
            print(f"Error in threading: {e}")
            face_match = False
    
    counter += 1

    if face_match:
        cv2.putText(frame, "MATCH", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO MATCH", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

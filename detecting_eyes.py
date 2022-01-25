import cv2
import dlib

cap = cv2.VideoCapture(0)		# input camera device number
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

color = (0,255,0)               # green
weight = 1

# right eye
lines = [(a, a + 1) for a in range(36, 41)]
lines += [(41,36)]

# left eye
lines += [(a, a + 1) for a in range(42, 47)]
lines += [(47,42)]

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Face
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, weight)

        # Eyes
        landmarks = predictor(gray, face)
        for (start,end) in lines:
            # Draw lines
            left_point = (landmarks.part(start).x, landmarks.part(start).y)
            right_point = (landmarks.part(end).x, landmarks.part(end).y)
            hor_line = cv2.line(frame, left_point, right_point, color, weight)

    # show result
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:       # Esc key
        break
        
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2

def main():
    cap =cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 5)
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                center_x = ex + (ew // 2)  # Calculate the x-coordinate of the center
                center_y = ey + (eh // 2)  # Calculate the y-coordinate of the center
                radius = min(ew, eh) // 2 
                #cv2.rectangle(roi_color, (ex,ey), (ex+ ew, ey + eh), (0, 0, 255), 5)
                cv2.circle(roi_color, (center_x, center_y), radius, (0, 0, 255), 5)


        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

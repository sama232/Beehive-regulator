from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('C:/Users/Nadine Mostafa/runs/detect/train5/weights/last.pt')

# load video
video_path = 'C:/Users/Nadine Mostafa/Desktop/bees vs wasps/Breathtaking Flight Maneuvers of Hornets in Slow Motion __ ViralHog.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        results = model.track(frame, persist=True)

        frame_ = results[0].plot()

        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

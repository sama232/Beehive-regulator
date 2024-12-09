from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('.pt') #copy and paste the .pt model's path (usually found in runs/detect/train/weights)

# load video
video_path = '.mp4' #copy and paste the video's path
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

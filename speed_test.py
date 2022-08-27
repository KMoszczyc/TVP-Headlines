from imutils.video import FileVideoStream
import time
import cv2

print("[INFO] starting video file thread...")
input_video_path = 'test_data/videos/vid1.mp4'
fvs = FileVideoStream(input_video_path, queue_size=1280).start()
cap = cv2.VideoCapture(input_video_path)
time.sleep(1.0)

start_time = time.time()

while fvs.more():
     # _, frame = cap.read()
     frame = fvs.read()

print("imutils elasped time: {:.2f}s".format(time.time() - start_time))
start_time = time.time()

while True:
     ret, frame = cap.read()
     # frame = fvs.read()

     if not ret:
         break


print("OpenCV elasped time: {:.2f}s".format(time.time() - start_time))
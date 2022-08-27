import cv2
import numpy as np
import pytesseract
import time
from FPS import FPS

# Tesseract Page segmentation modes (-- psm):
#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR.
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line. <-- THIS
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific. <-- OR THIS


pytesseract.pytesseract.tesseract_cmd = 'D:/Program Files/Tesseract-OCR/tesseract.exe'
print(pytesseract.get_languages(config=''))

HEADLINE_AVG_COLOR = (129.5148472, 62.9367192, 53.23520085)  # BGR
SECONDS_TO_SKIP_AFTER_LAST_HEADLINE = 20
HEADLINE_ANIMATION_SECONDS = 2
FRAMES_TO_SKIP = 75


def extract_headline(frame, do_ocr):
    """Extract headline from a single frame"""
    h, w, _ = frame.shape
    x1 = int(w * 0.17)
    x2 = int(w * 0.925)
    y1 = int(h * 0.78)
    y2 = int(h * 0.89)

    headline = ''
    headline_like_frame_detected = False
    headline_img = frame[y1:y2, x1:x2]

    # Get avg color and compare it to a headline avg color
    current_avg_color = np.average(np.average(headline_img, axis=0), axis=0)
    col_diff = abs(np.subtract(HEADLINE_AVG_COLOR, current_avg_color))

    # if bgr channels are less different than 10 (in scale 0-255) then it's probably a headline
    if col_diff[0] < 10 and col_diff[1] < 10 and col_diff[2] < 10:
        print('Color diff:', col_diff)
        headline_like_frame_detected = True
        if do_ocr:
            resize_scale = 0.4
            gray_img = cv2.cvtColor(headline_img, cv2.COLOR_BGR2GRAY)
            ret, binary_img = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
            resized_img = cv2.resize(binary_img, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
            headline = pytesseract.image_to_string(resized_img, config=r'--psm 7 ', lang='pol')

    return headline_img, headline, headline_like_frame_detected


def tvp_headlines_mp4(input_video_path, output_headlines_path):
    """Extract all headlines from a .mp4 file"""
    cap = cv2.VideoCapture(input_video_path)
    fps = FPS()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(width, height, video_fps)

    headlines = []
    frame_count = 0
    last_headline_frame = 0
    headline_animation_start_frame = 0
    has_headline_animation_count_started = False
    do_ocr = False

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # Skip ocr if last headline was seen 20s ago
        if frames_to_seconds(video_fps, frame_count - last_headline_frame) > SECONDS_TO_SKIP_AFTER_LAST_HEADLINE:

            # Wait for headline animation to end (2S)
            if frames_to_seconds(video_fps, frame_count - headline_animation_start_frame) > HEADLINE_ANIMATION_SECONDS and has_headline_animation_count_started:
                do_ocr = True

            # Extract headline with ocr
            headline_img, headline, headline_like_frame_detected = extract_headline(frame, do_ocr)

            # If extract_headline stopped detecting headline-like frames then stop animation counters (headline-like frames must be detected every frame for at least 2 seconds)
            if has_headline_animation_count_started and not headline_like_frame_detected:
                has_headline_animation_count_started = False

            # If headline-like frame was detected and animation counter hasn't started yet then start it
            if headline_like_frame_detected and not has_headline_animation_count_started:
                has_headline_animation_count_started = True
                headline_animation_start_frame = frame_count

            # Save headline, reset counters
            if headline != '':
                print('Frame:', frame_count, 'Headline:', headline)
                last_headline_frame = frame_count
                headlines.append(headline)
                do_ocr = False
                has_headline_animation_count_started = False
                headline_animation_start_frame = 0

                save_headline(headline, output_headlines_path)

        time_elapsed = frames_to_seconds(video_fps, frame_count)
        last_headline_elapsed = frames_to_seconds(video_fps, frame_count - last_headline_frame)
        animation_time_elapsed = 0 if headline_animation_start_frame == 0 else frames_to_seconds(video_fps, frame_count - headline_animation_start_frame)

        print('Time elapsed:', f'{time_elapsed}s', '\tLast headline:', f'{last_headline_elapsed}s',
              '\tHeadline animation:', f'{animation_time_elapsed}s', f'\tHeadline count: {len(headlines)}', f'FPS: {fps()}', frame_count)

        frame_count += 1

        # cv2.imshow('frame', frame)
        # c = cv2.waitKey(1)
        # if c & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    # fvs.stop()

    print(headlines)


def tvp_headlines_mp4_skip(input_video_path, output_headlines_path):
    """Extract all headlines from a .mp4 file"""

    cap = cv2.VideoCapture(input_video_path)
    fps = FPS()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(width, height, video_fps)

    headlines = []
    frame_count = 0
    last_headline_frame = 0
    headline_animation_start_frame = 0

    ocr_processing_times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip ocr if last headline was seen 20s ago
        if frames_to_seconds(video_fps, frame_count - last_headline_frame) > SECONDS_TO_SKIP_AFTER_LAST_HEADLINE:

            # Extract headline with ocr
            start_time = time.time()
            headline_img, headline, headline_like_frame_detected = extract_headline(frame, True)
            ocr_processing_times.append(time.time() - start_time)

            # Save headline, reset counters
            if headline != '':
                print('Frame:', frame_count, 'Headline:', headline)
                last_headline_frame = frame_count
                headlines.append(headline)
                headline_animation_start_frame = 0

                save_headline(headline, output_headlines_path)

        time_elapsed = frames_to_seconds(video_fps, frame_count)
        last_headline_elapsed = frames_to_seconds(video_fps, frame_count - last_headline_frame)
        animation_time_elapsed = 0 if headline_animation_start_frame == 0 else frames_to_seconds(video_fps, frame_count - headline_animation_start_frame)

        # Skip frames equal to FRAMES_TO_SKIP
        frame_count += FRAMES_TO_SKIP
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        print('Time elapsed:', f'{time_elapsed}s', '\tLast headline:', f'{last_headline_elapsed}s',
              '\tHeadline animation:', f'{animation_time_elapsed}s', f'\tHeadline count: {len(headlines)}', f'FPS: {fps()}', frame_count)

    cap.release()
    cv2.destroyAllWindows()

    print(headlines)
    print('OCR elapsed times:', [str(round(x, 3)) + 's' for x in sorted(ocr_processing_times, reverse=True)[:10]])


def frames_to_seconds(fps, frame_count):
    return frame_count / fps


def save_headline(headline, path):
    with open(path, 'a') as f:
        f.write(headline.strip('\n') + '\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_video_path = 'test_data/videos/vid1.mp4'
    output_headlines_path = 'headlines1.txt'

    start_time = time.time()
    # tvp_headlines_mp4(input_video_path, output_headlines_path)
    tvp_headlines_mp4_skip(input_video_path, output_headlines_path)

    print("elasped time: {:.2f}s".format(time.time() - start_time))

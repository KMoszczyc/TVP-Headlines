import os

import cv2
import numpy as np
import pytesseract
import time
from FPS import FPS
from video_download import download_video, download_all_videos
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

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


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
print(pytesseract.get_languages(config=''))

HEADLINE_AVG_COLOR = (129.5148472, 62.9367192, 53.23520085)  # BGR
SECONDS_TO_SKIP_AFTER_LAST_HEADLINE = 20
HEADLINE_ANIMATION_SECONDS = 3
FRAMES_TO_SKIP = 75
COLOR_SIMILARITY_THRESHOLD = 10
OCR_CONFIDENCE_THRESHOLD = 80  # [0-100]

dst_txt_headline_dir_path = 'output/txt_headlines'
dst_screenshot_headline_dir_path = 'output/screenshot_headlines'
last_headline_color = (0, 0, 0)


class HeadlineExtractor():
    def __init__(self):
        self.frame_count = 0
        self.last_headline_frame = 0
        self.headline_animation_start_frame = 0
        self.has_headline_animation_count_started = False
        self.headline_like_frame_detected = False
        self.do_ocr = False
        self.FPS = 0
        self.headline_screenshots = []
        self.headline_txts = []
        self.headlines = []
        self.ocr_processing_times = []

    def extract_headline(self, frame):
        """Extract headline from a single frame"""
        h, w, _ = frame.shape
        x1 = int(w * 0.17)
        x2 = int(w * 0.925)
        y1 = int(h * 0.78)
        y2 = int(h * 0.89)

        headline = ''
        self.headline_like_frame_detected = False
        headline_img = frame[y1:y2, x1:x2]

        # Get avg color and compare it to a headline avg color
        current_avg_color = np.average(np.average(headline_img, axis=0), axis=0)

        # if bgr channels are less different than the COLOR_SIMILARITY_THRESHOLD (10), (in scale 0-255) then it's probably a headline
        if self.are_colors_similar(HEADLINE_AVG_COLOR, current_avg_color, COLOR_SIMILARITY_THRESHOLD):
            self.headline_like_frame_detected = True
            if self.do_ocr:
                resize_scale = 0.9
                gray_img = cv2.cvtColor(headline_img, cv2.COLOR_BGR2GRAY)
                ret, binary_img = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
                resized_img = cv2.resize(binary_img, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)

                headline = self.ocr_on_img(resized_img)
                current_avg_color = tuple(int(x) for x in current_avg_color)

        return headline_img, headline, current_avg_color

    # TODO: 10.11.2023 brakujący "alarm: ucieczka spod buta łukaszenki - frame: 25164 time: 1006.56
    def ocr_on_img(self, img):
        df = pytesseract.image_to_data(img, lang='pol', config=r'--psm 7', output_type='data.frame')
        df_filtered = df[df['text'].notna()]

        print('frame:', self.frame_count, 'time:', self.frames_to_seconds(self.frame_count))
        # print('AAAAAAAAAAAAAA')
        #
        # print(df.head(50))
        # print('BBBBBBBBBBBBBBB')
        #
        # print(df_filtered.head(50))

        if df_filtered.empty:
            self.reset_animation_counters()
            return ''

        # print('CCCCCCCCCCCCCCCCC')
        df_filtered['text'] = df_filtered['text'].str.strip()
        is_correct = len(
            df_filtered[(df_filtered['conf'] < OCR_CONFIDENCE_THRESHOLD) | (df_filtered['text'].str.contains('\|')) | (df_filtered['text'] == '')]) == 0
        #
        # print(len(df_filtered[(df_filtered['conf'] < OCR_CONFIDENCE_THRESHOLD)]), len(df_filtered[df_filtered['text'].str.contains("\|")]), len(df_filtered[df_filtered['text'] == '']))
        # print(df_filtered.head(50))

        if is_correct:
            # headline = ' '.join(df_filtered['text'].to_list())
            headline = pytesseract.image_to_string(img, config=r'--psm 7', lang='pol')
        else:
            headline = ''
            self.reset_animation_counters()
        # print(is_correct, headline)

        return headline

    def calculate_color_diff(self, color_a, color_b):
        return abs(np.subtract(color_a, color_b))

    def are_colors_similar(self, color_a, color_b, threshold):
        color_diff = self.calculate_color_diff(color_a, color_b)
        return color_diff[0] <= threshold and color_diff[1] <= threshold and color_diff[2] <= threshold

    def reset_animation_counters(self):
        self.has_headline_animation_count_started = False
        self.headline_animation_start_frame = self.frame_count
        self.headline_like_frame_detected = False
        self.do_ocr = False

    def reset_all_counters(self):
        self.last_headline_frame = self.frame_count
        self.do_ocr = False
        self.has_headline_animation_count_started = False
        self.headline_animation_start_frame = 0

    def tvp_headlines_mp4(self, input_video_path, output_headlines_path, dst_screenshot_headlines_path):
        """Extract all headlines from a .mp4 file"""
        global last_headline_color
        cap = cv2.VideoCapture(input_video_path)
        fps = FPS()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = cap.get(cv2.CAP_PROP_FPS)

        print(width, height, self.FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip ocr if last headline was seen 20s ago
            if self.frames_to_seconds(self.frame_count - self.last_headline_frame) > SECONDS_TO_SKIP_AFTER_LAST_HEADLINE:

                # Wait for headline animation to end (2S) - makes sure that we don't OCR on an empty headline or that we skip short headlines with currently presented person names
                if self.frames_to_seconds(self.frame_count - self.headline_animation_start_frame) > HEADLINE_ANIMATION_SECONDS and self.has_headline_animation_count_started:
                    self.do_ocr = True

                # Extract headline with ocr
                headline_img, headline, headline_avg_color = self.extract_headline(frame)

                # If headline-like frame was detected and animation counter hasn't started yet then start it
                if self.headline_like_frame_detected and not self.has_headline_animation_count_started:
                    self.has_headline_animation_count_started = True
                    self.headline_animation_start_frame = self.frame_count

                # If extract_headline stopped detecting headline-like frames then stop animation counters (headline-like frames must be detected every frame for at least 2 seconds)
                if self.has_headline_animation_count_started and not self.headline_like_frame_detected:
                    self.has_headline_animation_count_started = False
                    self.headline_animation_start_frame = self.frame_count

                if headline != '' and headline in self.headline_txts:
                    self.reset_all_counters()
                elif headline != '':  # Save headline, reset counters
                    print('Frame:', self.frame_count, 'Headline:', headline)
                    print(last_headline_color, headline_avg_color)
                    last_headline_color = headline_avg_color

                    self.reset_all_counters()
                    self.headline_txts.append(headline)
                    self.headline_screenshots.append(headline_img)
                    self.save_headline(headline, output_headlines_path)

            time_elapsed = self.frames_to_seconds(self.frame_count)
            last_headline_elapsed = self.frames_to_seconds(self.frame_count - self.last_headline_frame)
            animation_time_elapsed = 0 if self.headline_animation_start_frame == 0 else self.frames_to_seconds(
                self.frame_count - self.headline_animation_start_frame)

            if self.frame_count % self.seconds_to_frames(10) == 0:
                print('Time elapsed:', f'{time_elapsed}s', '\tLast headline:', f'{last_headline_elapsed}s',
                      '\tHeadline animation:', f'{animation_time_elapsed}s', f'\tHeadline count: {len(self.headline_txts)}', f'FPS: {fps()}', self.frame_count)

            self.frame_count += 1

            # cv2.imshow('frame', frame)
            # c = cv2.waitKey(1)
            # if c & 0xFF == ord('q'):
            #     break

        # Save merged headline screenshots
        screenshots_merged = np.concatenate(self.headline_screenshots, axis=0)
        cv2.imwrite(dst_screenshot_headlines_path, screenshots_merged)

        cap.release()
        cv2.destroyAllWindows()
        # fvs.stop()

        print(self.headline_txts)

    def tvp_headlines_mp4_skip(self, input_video_path, output_headlines_path, video_name):
        """Extract all headlines from a .mp4 file"""

        cap = cv2.VideoCapture(input_video_path)
        fps = FPS()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        print(width, height, video_fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip ocr if last headline was seen 20s ago
            if self.frames_to_seconds(self.frame_count - self.last_headline_frame) > SECONDS_TO_SKIP_AFTER_LAST_HEADLINE:

                # Extract headline with ocr
                start_time = time.time()
                headline_img, headline, headline_like_frame_detected = self.extract_headline(frame, True)
                self.ocr_processing_times.append(time.time() - start_time)

                # Save headline, reset counters
                if headline != '':
                    print('Frame:', self.frame_count, 'Headline:', headline)
                    self.last_headline_frame = self.frame_count
                    self.headline_txts.append(headline)
                    self.headline_animation_start_frame = 0

                    self.save_headline(headline, output_headlines_path)

            time_elapsed = self.frames_to_seconds(self.frame_count)
            last_headline_elapsed = self.frames_to_seconds(self.frame_count - self.last_headline_frame)
            animation_time_elapsed = 0 if self.headline_animation_start_frame == 0 else self.frames_to_seconds(self.frame_count - self.headline_animation_start_frame)

            # Skip frames equal to FRAMES_TO_SKIP
            self.frame_count += FRAMES_TO_SKIP
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

            if self.frame_count % self.seconds_to_frames(10) == 0:
                print('Time elapsed:', f'{time_elapsed}s', '\tLast headline:', f'{last_headline_elapsed}s',
                      '\tHeadline animation:', f'{animation_time_elapsed}s', f'\tHeadline count: {len(self.headline_txts)}', f'FPS: {fps()}', self.frame_count)

        cap.release()
        cv2.destroyAllWindows()

        print(self.headline_txts)
        print('OCR elapsed times:', [str(round(x, 3)) + 's' for x in sorted(self.ocr_processing_times, reverse=True)[:10]])

    def frames_to_seconds(self, frame_count):
        return frame_count / self.FPS

    def seconds_to_frames(self, seconds):
        return self.FPS * seconds

    def save_headline(self, headline, path):
        with open(path, 'a') as f:
            f.write(headline.strip('\n') + '\n')

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} is created!")

    def process_video(self, input_video_path):
        video_name = os.path.basename(input_video_path).split('.mp4')[0]
        dst_txt_headlines_path = os.path.join(dst_txt_headline_dir_path, f'{video_name}.txt')
        dst_screenshot_headlines_path = os.path.join(dst_screenshot_headline_dir_path, f'{video_name}.jpg')

        start_time = time.time()
        self.tvp_headlines_mp4(input_video_path, dst_txt_headlines_path, dst_screenshot_headlines_path)

        print("elasped time: {:.2f}s".format(time.time() - start_time))

    def process_videos(self):
        video_dir_path = 'data/'
        video_filenames = [filename for filename in os.listdir(video_dir_path) if filename.rsplit('.', 1)[1] == 'mp4']
        dst_txt_headline_dir_path = 'output/txt_headlines'
        dst_screenshot_headline_dir_path = 'output/screenshot_headlines'

        # create dirs if they don't exist
        self.create_dir(video_dir_path)
        self.create_dir(dst_txt_headline_dir_path)
        self.create_dir(dst_screenshot_headline_dir_path)

        for filename in video_filenames:
            video_name = filename.split('.mp4')[0]
            input_video_path = os.path.join(video_dir_path, filename)
            dst_txt_headlines_path = os.path.join(dst_txt_headline_dir_path, f'{video_name}.txt')
            dst_screenshot_headlines_path = os.path.join(dst_screenshot_headline_dir_path, f'{video_name}.jpg')

            start_time = time.time()
            self.tvp_headlines_mp4(input_video_path, dst_txt_headlines_path, dst_screenshot_headlines_path)
            # self.tvp_headlines_mp4_skip(input_video_path, dst_txt_headlines_path)

            print("elasped time: {:.2f}s".format(time.time() - start_time))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # download_video()
    # download_all_videos()

    extractor = HeadlineExtractor()
    # extractor.process_videos()

    input_video_path = 'data/10.11.2023, 19_30.mp4'
    extractor.process_video(input_video_path)

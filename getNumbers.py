import cv2
from datetime import datetime, time
import numpy as np
import imutils
from imutils import contours
import glob
import re
from statistics import mode
import json

class SevenSegmentRecognizer:

    GREEN = (0,255,0)
    RED = (0, 0, 255)

    TRAINING_IMG_SIZE = (166, 100) #(163, 199)

    HUNDREDS_RANGE = (5, 48)
    TENS_RANGE = (49, 75)
    ONES_RANGE = (76, 96)
    TENTH_RANGE = (97, 119)
    HUNDREDTH_RANGE = (120, 150)

    RANGES = [HUNDREDS_RANGE, TENS_RANGE, ONES_RANGE, TENTH_RANGE, HUNDREDTH_RANGE]

    RANGE_MAP = {
        0 : 100,
        1 : 10,
        2 : 1,
        3 : 0.1,
        4 : 0.01,
    }

    # define the dictionary of digit segments so we can identify
    # each digit on the thermostat
    _DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9,
        (0, 0, 1, 1, 0, 0, 0): '.',
    }

    def __init__(self):
        self._frame_info = {}
        self._frame_no = 0
        self._first_frame = True
        self._camera_indx = 0
        self._date = datetime.today()
        self._window_name = "self._curr_frame"

        self._num_corners = 0
        self._corners = []
        self._got_corners = False

        self._persisten_drawings = []

        self._counter = 0

        self._curr_num = 0

    def _draw_persistents(self):
        for drawing in self._persisten_drawings:
            func = getattr(cv2, drawing['type'])
            args = [self._curr_frame] + drawing['args']
            self._curr_frame = func(*args)

    def _get_perspective_transform(self):
        pt_A = self._corners[0]
        pt_B = self._corners[1]
        pt_C = self._corners[2]
        pt_D = self._corners[3]

        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        self._max_width = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        self._maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, self._maxHeight - 1],
                                [self._max_width - 1, self._maxHeight - 1],
                                [self._max_width - 1, 0]])

        self._M = cv2.getPerspectiveTransform(input_pts,output_pts)


    def _get_lcd_corners(self, event, x, y, param, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = {
                'type' : 'circle',
                'args' : [(x,y), 2, self.GREEN, 3]
            }
            self._persisten_drawings.append(drawing)
            self._corners.append([x,y])
            print("Clicked: ", x, y)
            self._num_corners += 1

        if event == cv2.EVENT_RBUTTONDOWN:
            self._persisten_drawings.pop()
            self._corners.pop()
            self._num_corners -= 1

        if self._num_corners == 4:
            self._persisten_drawings = []
            self._got_corners = True
            self._get_perspective_transform()


    def _handle_ts(self, ts):
        # # For some reason, the first frame ts does not get logged as zero
        # if self._first_frame:
        #     self._og_ts = ts
        #     self._first_frame = False

        # ts -= self._og_ts

        # Creating dictionary with ts
        if self._frame_exists:
            if self._frame_no not in self._frame_info:
                self._frame_info[self._frame_no] =  {'timestamp': ts}

    def _look_for_digits(self):
        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(self._curr_frame, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)


        cnts = imutils.grab_contours(cnts)
        self._digitCnts = []
        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            if w >= 8 and (h >= 30 and h <= 70):
                self._digitCnts.append(c)

        print(self._digitCnts)
        cv2.drawContours(self._out_frame, self._digitCnts, -1, self.GREEN, 1)

        # sort the contours from left-to-right, then initialize the
        # actual digits themselves
        self._digitCnts = contours.sort_contours(self._digitCnts,
            method="left-to-right")[0]
        digits = []

        # loop over each of the digits
        for c in self._digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            roi = self._curr_frame[y:y + h, x:x + w]
            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)
            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),	# top
                ((0, 0), (dW, h // 2)),	# top-left
                ((w - dW, 0), (w, h // 2)),	# top-right
                ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
                ((0, h // 2), (dW, h)),	# bottom-left
                ((w - dW, h // 2), (w, h)),	# bottom-right
                ((0, h - dH), (w, h))	# bottom
            ]
            on = [0] * len(segments)

        	# loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) > 0.5:
                    on[i]= 1
            # lookup the digit and draw it on the image
            try:
                digit = self._DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
                cv2.rectangle(self._out_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(self._out_frame, str(digit), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            except:
                pass

    def _find_template_digits(self):
        ttype = 'binary'
        template_files = glob.glob(f'./template_images/*_{ttype}.png')
        curr_num = 0

        taken_digits = [0, 0, 0, 0, 0]
        digit_history = [[0],[0],[0],[0],[0]]

        for template_file in template_files:
            num = re.findall('[0-9]', template_file)[0]
            template = cv2.imread(template_file,0)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(self._curr_frame,template,cv2.TM_CCORR_NORMED)
            threshold = 0.80
            loc = np.where( res >= threshold)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(self._out_frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
                x = pt[0]
                mid_x = x + w//2

                for indx, base in enumerate(self.RANGES):
                    if (mid_x > base[0]) and (mid_x < base[1]):
                        digit_history[indx].append(int(num))
                        # if taken_digits[indx] == 0:
                        #     curr_num += self.RANGE_MAP[indx] * int(num)
                        #     taken_digits[indx] = 1
                        break

        if not self._counter%10:
            for indx, digits in enumerate(digit_history):
                curr_num += self.RANGE_MAP[indx] * mode(digits)

            print(curr_num)
            self._persisten_drawings = []
            drawing = {
                'type' : 'putText',
                'args' : [str(curr_num), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1]
            }
            self._persisten_drawings.append(drawing)
            self._curr_num = curr_num

    def _filter_image(self):
         # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        self._curr_frame = cv2.cvtColor(self._curr_frame, cv2.COLOR_BGR2GRAY)
        self._curr_frame = cv2.threshold(self._curr_frame, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
        self._curr_frame = cv2.morphologyEx(self._curr_frame, cv2.MORPH_OPEN, kernel)

    def _get_numbers(self):
        self._filter_image()
        # self._look_for_digits()
        self._find_template_digits()
        self._frame_info[self._frame_no]['wattage'] = self._curr_num

    def run(self):
        print("Running")

        cap = cv2.VideoCapture(self._camera_indx)

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name,self._get_lcd_corners)

        while True:
            self._frame_exists, self._curr_frame = cap.read()
            self._full_frame = self._curr_frame.copy()
            ts = str(datetime.now())#cap.get(cv2.CAP_PROP_POS_MSEC)


            # Once corners are determined, we don't need the callback
            if self._got_corners:
                cv2.setMouseCallback(self._window_name, lambda *args : None)
                self._curr_frame = cv2.warpPerspective(self._curr_frame,self._M,(self._max_width, self._maxHeight),flags=cv2.INTER_LINEAR)

                self._curr_frame = cv2.resize(self._curr_frame, self.TRAINING_IMG_SIZE)
                self._out_frame = self._curr_frame.copy()
                self._counter += 1
                self._handle_ts(ts)
                try:
                    self._get_numbers()
                    cv2.imwrite(f'binary_images/f{self._frame_no}.jpg', self._curr_frame)
                except Exception as e:
                    print(e)
                self._frame_no += 1


                cv2.imshow('processing', self._curr_frame.copy())
                self._curr_frame = self._out_frame

            self._draw_persistents()

            cv2.imshow(self._window_name, self._curr_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(self._frame_info)
                with open(f'wattage_info_{datetime.now()}.json', 'w') as f:
                    json.dump(self._frame_info,f, indent=6)
                break


if __name__ == '__main__':
    proc = SevenSegmentRecognizer()
    proc.run()
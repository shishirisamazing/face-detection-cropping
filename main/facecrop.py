#author: Tomo Lapautre

import os
import warnings

# Suppress noisy warnings before importing heavy libraries
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')           # TensorFlow: suppress INFO + WARNING
os.environ.setdefault('GLOG_minloglevel', '2')               # MediaPipe: suppress INFO + WARNING

# Suppress pymatting Cholesky decomposition warnings (triggered by alpha_matting on some images)
warnings.filterwarnings('ignore', message='.*Cholesky.*', category=UserWarning)

# Import rembg before cv2/mediapipe to avoid onnxruntime conflicts
try:
    from rembg import remove, new_session
    _REMBG_AVAILABLE = True
except (ImportError, SystemExit):
    _REMBG_AVAILABLE = False

import cv2
import numpy as np
import mediapipe as mp

import pathlib
from pathlib import Path
import glob
from PIL import Image, ImageOps

from main.constants import (
    PNGS,
    CV2_FILETYPES,
    DNN_THRESHOLD,
    FAILED_IMAGE_FOLDER)

# Resolve model paths relative to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FACE_MODEL_PATH = os.path.join(_THIS_DIR, 'blaze_face_short_range.tflite')


class FaceCrop():

    def __init__(self, height=0, width=0, height_asy=0, width_asy=0, tag='A', pyqt_ui=None, bg_color=None,
                 mode='percentage', aspect_ratio=(1, 1), padding_multiplier=2.5,
                 custom_width_px=600, custom_height_px=800):

        self.width = width
        self.width_asy = width_asy
        self.height = height
        self.height_asy = height_asy
        self.tag = tag
        self.mode = mode
        self.aspect_ratio = aspect_ratio
        self.padding_multiplier = padding_multiplier
        self.custom_width_px = custom_width_px
        self.custom_height_px = custom_height_px
        self.failure_folder = FAILED_IMAGE_FOLDER #folder where images where faces couldn't be detected will be stored
        self.threshold = DNN_THRESHOLD #threshold of confidence to categorise as a face
        self.progress_count = 0 #to keep track of progress bar
        self.pyqt_ui = pyqt_ui #this variable is to integrate this class to your pyqt GUI so that, for example you can generate a progress bar
        self.bg_color = bg_color #tuple (R, G, B) or None to skip background replacement

        # Initialize MediaPipe Face Detection (tasks API)
        BaseOptions = mp.tasks.BaseOptions
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self.face_detection = mp.tasks.vision.FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=_FACE_MODEL_PATH),
                min_detection_confidence=self.threshold
            )
        )

        # Initialize rembg session for background removal if enabled
        self._rembg_session = None
        if self.bg_color is not None and _REMBG_AVAILABLE:
            self._rembg_session = new_session("isnet-general-use",
                                              providers=["CPUExecutionProvider"])


    def _compute_crop_px(self, img_width, img_height, face_w=None, face_h=None):
        """Compute crop dimensions in pixels based on the selected mode."""
        if self.mode == 'percentage':
            width_px = int((abs(self.width) * img_width) / 100)
            height_px = int((abs(self.height) * img_height) / 100)
        elif self.mode == 'aspect_ratio':
            ratio_w, ratio_h = self.aspect_ratio
            mult = self.padding_multiplier
            crop_w = max(face_w * mult, face_h * mult * ratio_w / ratio_h)
            crop_h = crop_w * ratio_h / ratio_w
            width_px = min(int(crop_w), img_width)
            height_px = min(int(crop_h), img_height)
        elif self.mode == 'custom_pixels':
            width_px = int(self.custom_width_px)
            height_px = int(self.custom_height_px)
        else:
            raise ValueError("Unknown crop mode: {}".format(self.mode))
        return width_px, height_px


    #finds the face and saves the cropped face in your output directory
    def crop_save(self, input_directory, output_path, bool_folder=False, bool_face_count=False, preview=False, excluded_files=None):
        #if 'bool_face_count' is set to True, program will only save one face per image (the one with the highest confidence)
        #if 'bool_folder' is set to True, program will save the new image in its seperate folder. This can be useful if you have multiple faces per image.
        #if 'preview' is set to True, only crops the first face it can find and returns the image.

        files = glob.glob('{}/*'.format(input_directory)) #finds all the files in your directory

        #loops through all the files in the directory
        for i, file in enumerate(files):

            #updates the pyqt progress bar
            if not preview and self.pyqt_ui is not None:
                self.progress_count += 1
                self.pyqt_ui.progress.setValue(self.progress_count)

                #breaks program if user clicks on cancel button of progress bar
                if self.pyqt_ui.progress.wasCanceled():
                    break


            file_path = Path(file)
            file_name = file_path.stem
            ext = file_path.suffix

            # Skip files excluded by user
            if excluded_files and file in excluded_files:
                continue

            #checks if image is readable by cv2
            if ext.lower() not in CV2_FILETYPES:
                continue

            image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)

            try:
                img_height, img_width = image.shape[:2]
            except AttributeError:
                print('{}: ImageReadError'.format(file_name))
                continue

            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = self.face_detection.detect(mp_image)

            #reload image using PIL as cv2 isn't adapted in this scenario to read PNGs properly. This increases the length of operation but is the only solution I could find.
            #(cv2 can read pngs by specifying cv2.IMREAD_UNCHANGED but this is a double edged sword as you lose the EXIF data of the image as well)
            temp_file = np.asarray(ImageOps.exif_transpose(Image.open(file)))

            k = 0 #'k' keeps track of how faces pass the threshold test in an image

            # Build list of detections sorted by confidence (highest first)
            detections = []
            for detection in results.detections:
                score = detection.categories[0].score
                if score > self.threshold:
                    detections.append(detection)
            detections.sort(key=lambda d: d.categories[0].score, reverse=True)

            if bool_face_count: #only saves the face with the highest confidence
                detections = detections[:1]

            #loop through all the detected faces
            for det in detections:
                k+=1

                # Extract bounding box from MediaPipe detection (pixel coordinates)
                bbox = det.bounding_box
                x0 = max(0, bbox.origin_x)
                y0 = max(0, bbox.origin_y)
                x1 = min(img_width, bbox.origin_x + bbox.width)
                y1 = min(img_height, bbox.origin_y + bbox.height)

                h = y1 - y0 #height of face
                w = x1 - x0 #width of face

                width_px, height_px = self._compute_crop_px(img_width, img_height, face_w=w, face_h=h)

                south = min(int(y0 + 0.5*h + ((50+self.height_asy)/100)*height_px), img_height) #southern border of our new cropped image

                #makes sure that north and south are not outside the original image
                if south - height_px < 0 and ext.lower() not in PNGS:
                    south = height_px
                    north = 0
                else:
                    north = max(south - height_px, 0)

                west = max(int(x0 + 0.5*w - ((50+self.width_asy)/100)*width_px), 0) #western border of our new cropped image

                #makes sure that east and west are not outside the original image
                if west + width_px > img_width:
                    west = img_width - width_px
                    east = img_width
                else:
                    east = min(west + width_px, img_width)

                #crops the new image from the original one
                face  = temp_file[north : south, west : east]


                #if image is a PNG, we can add extend the height above the top of the head to make sure every cropped image has the same space above their head.
                #This is useful if you have images of tall individuals with very little space between the top of his/her head
                if ext.lower() in PNGS:
                    extra_height = height_px - min(int(y0 + 0.5*h + ((50+self.height_asy)/100)*height_px), img_height)
                    if extra_height > 0:
                        extra_layer = np.full((extra_height, face.shape[1], 4), 255, dtype='uint8')
                        extra_layer[:, :, 3] = 0 #value of 0 is a white background
                        face = np.concatenate((extra_layer, face), axis=0)

                cropped_face = Image.fromarray(face)

                #replace background with solid color if enabled
                if self.bg_color is not None and self._rembg_session is not None:
                    cropped_face = self._replace_background(face)

                #only looks for the first face it can find if in preview mode
                if preview:
                    return cropped_face


                file_name_folder = file_name.rstrip()

                #saves the cropped image
                if bool_folder:
                    if not os.path.exists('{0}/{1}'.format(output_path, str(file_name_folder))): #checks if directory already exists
                        os.makedirs('{0}/{1}'.format(output_path, str(file_name_folder)))
                    if k==1:
                        cropped_face.save('{0}/{1}/{1}{2}'.format(output_path, str(file_name_folder), ext))
                    else:
                        cropped_face.save('{0}/{1}/{1}_{2}{3}'.format(output_path, str(file_name_folder), k, ext))

                else:
                    if k==1:
                        cropped_face.save('{0}/{1}{2}'.format(output_path, str(file_name), ext))
                    else:
                        cropped_face.save('{0}/{1}_{2}{3}'.format(output_path, str(file_name), k, ext))

            #checks if program couldn't find any face for an image. If so, will save the original image in a seperate folder
            if k == 0:
                if preview:
                    pass

                else:
                    print('{}: Failed to detect face'.format(file_name))
                    if not os.path.exists('{0}/{1}'.format(output_path, self.failure_folder)):
                        os.mkdir('{0}/{1}'.format(output_path, self.failure_folder))
                    Image.fromarray(temp_file).save('{0}/{1}/{2}{3}'.format(output_path, self.failure_folder, str(file_name), ext))

                continue

    def crop_single(self, file_path):
        """Crop a single image and return the PIL Image, or None if no face found."""
        file_path = str(file_path)
        fp = Path(file_path)
        ext = fp.suffix

        if ext.lower() not in CV2_FILETYPES:
            return None

        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        try:
            img_height, img_width = image.shape[:2]
        except AttributeError:
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self.face_detection.detect(mp_image)

        temp_file = np.asarray(ImageOps.exif_transpose(Image.open(file_path)))

        detections = []
        for detection in results.detections:
            score = detection.categories[0].score
            if score > self.threshold:
                detections.append(detection)
        detections.sort(key=lambda d: d.categories[0].score, reverse=True)

        if not detections:
            return None

        det = detections[0]

        # Bounding box in pixel coordinates
        bbox = det.bounding_box
        x0 = max(0, bbox.origin_x)
        y0 = max(0, bbox.origin_y)
        x1 = min(img_width, bbox.origin_x + bbox.width)
        y1 = min(img_height, bbox.origin_y + bbox.height)

        h = y1 - y0
        w = x1 - x0

        width_px, height_px = self._compute_crop_px(img_width, img_height, face_w=w, face_h=h)

        south = min(int(y0 + 0.5*h + ((50+self.height_asy)/100)*height_px), img_height)

        if south - height_px < 0 and ext.lower() not in PNGS:
            south = height_px
            north = 0
        else:
            north = max(south - height_px, 0)

        west = max(int(x0 + 0.5*w - ((50+self.width_asy)/100)*width_px), 0)

        if west + width_px > img_width:
            west = img_width - width_px
            east = img_width
        else:
            east = min(west + width_px, img_width)

        face = temp_file[north:south, west:east]

        if ext.lower() in PNGS:
            extra_height = height_px - min(int(y0 + 0.5*h + ((50+self.height_asy)/100)*height_px), img_height)
            if extra_height > 0:
                extra_layer = np.full((extra_height, face.shape[1], 4), 255, dtype='uint8')
                extra_layer[:, :, 3] = 0
                face = np.concatenate((extra_layer, face), axis=0)

        cropped_face = Image.fromarray(face)

        if self.bg_color is not None and self._rembg_session is not None:
            cropped_face = self._replace_background(face)

        return cropped_face

    def _replace_background(self, face_array):
        """Replace background with solid color using rembg (IS-Net model).

        rembg produces a high-quality alpha matte that handles fine hair detail,
        semi-transparent edges, and complex boundaries naturally.
        """
        # Ensure 3-channel RGB
        if face_array.shape[2] == 4:
            rgb_face = face_array[:, :, :3]
        else:
            rgb_face = face_array

        # Convert to PIL for rembg
        pil_input = Image.fromarray(rgb_face)

        # rembg returns RGBA image with alpha channel as the matte
        pil_rgba = remove(
            pil_input,
            session=self._rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )

        rgba_array = np.array(pil_rgba)
        alpha = rgba_array[:, :, 3].astype(np.float32) / 255.0
        fg = rgba_array[:, :, :3]

        # Alpha-blend foreground onto the chosen solid background color
        bg = np.full_like(fg, self.bg_color, dtype=np.uint8)
        alpha_3d = alpha[:, :, np.newaxis]
        output = (fg.astype(np.float32) * alpha_3d +
                  bg.astype(np.float32) * (1.0 - alpha_3d))
        output = np.clip(output, 0, 255).astype(np.uint8)

        return Image.fromarray(output)

    def close(self):
        """Release MediaPipe resources."""
        self.face_detection.close()

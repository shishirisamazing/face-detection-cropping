# FaceCrop

Automatic face detection and cropping tool with a modern dark-themed GUI. Detects faces in images using MediaPipe and crops them according to configurable settings.

## Features

- **Face Detection** powered by MediaPipe (BlazeFace model)
- **Three Crop Modes**:
  - **Percentage** - crop by percentage of the original image dimensions
  - **Aspect Ratio** - crop to a specific aspect ratio (1:1, 3:4, 2:3, 5:7, or custom) with adjustable padding
  - **Custom Pixels** - crop to exact pixel dimensions
- **Batch Processing** - process entire folders of images at once
- **Gallery View** - browse and preview images with thumbnails; exclude individual files before cropping
- **Live Preview** - see crop results before committing
- **Background Replacement** - optionally replace the background with a solid colour (requires `rembg`)
- **Asymmetric Offsets** - fine-tune the crop position horizontally and vertically
- **Single/Multiple Face Mode** - choose to keep only the highest-confidence face or crop all detected faces
- **Subfolder Support** - optionally organise output into per-image folders
- **Bilingual UI** - English and French

## Supported Image Formats

BMP, DIB, JP2, JPE, JPEG, JPG, PBM, PGM, PNG, PPM, RAS, SR, TIF, TIFF, WEBP

## Installation

### Prerequisites

- Python 3.8 or later

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd face-detection-cropping

# Install dependencies
pip install -r requirements.txt

# (Optional) Install background removal support
pip install rembg onnxruntime
```

## Usage

### Running from Source

```bash
python run.py
```

### Building a Standalone Executable

```bash
pip install pyinstaller
pyinstaller run.spec
```

The executable will be created in `dist/FaceCrop/`.

## How It Works

1. **Select Input Folder** - choose a folder containing images
2. **Select Output Folder** - choose where cropped images will be saved
3. **Configure Crop Settings** - pick a crop mode and adjust parameters in the sidebar
4. **Preview** - click an image thumbnail to see a live crop preview
5. **Crop** - click the Crop button to batch-process all selected images

## Project Structure

```
face-detection-cropping/
  run.py                  # Entry point
  run.spec                # PyInstaller build spec
  requirements.txt        # Python dependencies
  public/
    logo.png              # Application icon
  main/
    __init__.py
    __main__.py           # GUI (PyQt5) and application logic
    facecrop.py           # Face detection and cropping engine
    constants.py          # Shared constants
    parameters.json       # Saved user preferences
    blaze_face_short_range.tflite  # MediaPipe face detection model
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

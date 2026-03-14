# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules, data files, and binaries for packaged libraries
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')
onnx_datas, onnx_binaries, onnx_hiddenimports = collect_all('onnxruntime')

# Collect rembg data files and submodules separately to avoid onnx.reference crash
rembg_datas = collect_data_files('rembg')
rembg_hiddenimports = collect_submodules('rembg')

all_datas = mp_datas + rembg_datas + onnx_datas
all_binaries = mp_binaries + onnx_binaries
all_hiddenimports = (mp_hiddenimports + rembg_hiddenimports + onnx_hiddenimports
                     + ['filetype', 'pooch'])

# Use .icns on macOS, .ico on Windows, .png as fallback
if sys.platform == 'darwin' and os.path.exists('public/logo.icns'):
    icon_file = 'public/logo.icns'
elif sys.platform == 'win32' and os.path.exists('public/logo.ico'):
    icon_file = 'public/logo.ico'
else:
    icon_file = 'public/logo.png'

a = Analysis(['run.py'],
             pathex=['.'],
             binaries=all_binaries,
             datas=[
                 ('main/parameters.json', 'main'),
                 ('main/blaze_face_short_range.tflite', 'main'),
                 ('public/logo.png', 'public'),
             ] + all_datas,
             hiddenimports=all_hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=['onnx.reference'],
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='FaceCrop',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          icon=icon_file)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='FaceCrop')

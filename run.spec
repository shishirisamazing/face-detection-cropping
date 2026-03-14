# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect all mediapipe submodules, data files, and binaries
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')

# Use .icns on macOS, .ico on Windows, .png as fallback
if sys.platform == 'darwin' and os.path.exists('public/logo.icns'):
    icon_file = 'public/logo.icns'
elif sys.platform == 'win32' and os.path.exists('public/logo.ico'):
    icon_file = 'public/logo.ico'
else:
    icon_file = 'public/logo.png'

a = Analysis(['run.py'],
             pathex=['.'],
             binaries=mp_binaries,
             datas=[
                 ('main/parameters.json', 'main'),
                 ('main/blaze_face_short_range.tflite', 'main'),
                 ('public/logo.png', 'public'),
             ] + mp_datas,
             hiddenimports=mp_hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
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

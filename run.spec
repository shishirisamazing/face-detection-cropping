# -*- mode: python ; coding: utf-8 -*-
import sys

block_cipher = None

# Use .icns on macOS, .ico on Windows, .png as fallback
if sys.platform == 'darwin':
    icon_file = 'public/logo.icns'
elif sys.platform == 'win32':
    icon_file = 'public/logo.ico'
else:
    icon_file = 'public/logo.png'

a = Analysis(['run.py'],
             pathex=['.'],
             binaries=[],
             datas=[
                 ('main/parameters.json', 'main'),
                 ('main/blaze_face_short_range.tflite', 'main'),
                 ('public/logo.png', 'public'),
             ],
             hiddenimports=['mediapipe'],
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

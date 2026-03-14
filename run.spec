# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect all submodules, data files, and binaries for mediapipe (works fine)
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')

# Collect onnxruntime and rembg files (including .py sources) WITHOUT adding
# them to the module graph.  PyInstaller's binary-dependency scanner imports
# every collected package in an isolated child process; on Windows CI the
# onnxruntime native DLL (onnxruntime_pybind11_state.pyd) crashes with an
# access violation during that import.  By excluding these packages from
# Analysis and shipping their raw source files + binaries as data, we sidestep
# the child-process crash while keeping everything available at runtime.
onnx_datas = collect_data_files('onnxruntime', include_py_files=True)
onnx_binaries = collect_dynamic_libs('onnxruntime')

rembg_datas = collect_data_files('rembg', include_py_files=True)
rembg_binaries = collect_dynamic_libs('rembg')

all_datas = mp_datas + rembg_datas + onnx_datas
all_binaries = mp_binaries + onnx_binaries + rembg_binaries
# pymatting is a rembg dependency; since rembg is excluded from analysis its
# dependency tree isn't traced, so we list it here explicitly.
all_hiddenimports = (mp_hiddenimports
                     + ['filetype', 'pooch', 'pymatting'])

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
             excludes=['onnx.reference', 'onnxruntime', 'rembg'],
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

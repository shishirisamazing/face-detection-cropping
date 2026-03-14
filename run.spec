# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# ---- Work around onnxruntime crash during binary dependency scan ----------
# PyInstaller 6.x imports every collected package in an isolated child process
# to discover DLL dependencies (find_binary_dependencies).  On Windows CI,
# onnxruntime_pybind11_state.pyd crashes with an access violation during that
# import.  Patch the scanner to skip onnxruntime and rembg (which imports
# onnxruntime at module level).  Their native DLLs are collected explicitly
# via collect_dynamic_libs below, so the scan is unnecessary for them.
import PyInstaller.building.build_main as _build_main
_orig_find_binary_deps = _build_main.find_binary_dependencies

def _patched_find_binary_deps(binaries, collected_packages, *args, **kwargs):
    _skip = {'onnxruntime', 'rembg'}
    filtered = [p for p in collected_packages if p.split('.')[0] not in _skip]
    return _orig_find_binary_deps(binaries, filtered, *args, **kwargs)

_build_main.find_binary_dependencies = _patched_find_binary_deps
# --------------------------------------------------------------------------

# Collect all submodules, data files, and binaries for mediapipe
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')

# Collect onnxruntime and rembg data/submodules/binaries separately
onnx_datas = collect_data_files('onnxruntime')
onnx_hiddenimports = collect_submodules('onnxruntime')
onnx_binaries = collect_dynamic_libs('onnxruntime')

rembg_datas = collect_data_files('rembg')
rembg_hiddenimports = collect_submodules('rembg')

# collect_submodules('rembg') often fails to discover session submodules because
# importing rembg triggers onnxruntime which crashes on Windows CI.  Explicitly
# list the core rembg modules so they are always bundled.  new_session() relies
# on BaseSession.__subclasses__(), which only works if the session modules have
# actually been imported.
_rembg_explicit = [
    'rembg.bg',
    'rembg.sessions',
    'rembg.sessions.base',
    'rembg.sessions.isnet_general_use',
    'rembg.sessions.u2net',
    'rembg.sessions.u2netp',
    'rembg.sessions.u2net_human_seg',
    'rembg.sessions.u2net_cloth_seg',
    'rembg.sessions.silueta',
]

all_datas = mp_datas + rembg_datas + onnx_datas
all_binaries = mp_binaries + onnx_binaries
all_hiddenimports = (mp_hiddenimports + rembg_hiddenimports + onnx_hiddenimports
                     + _rembg_explicit
                     + ['filetype', 'pooch', 'pymatting', 'scipy'])

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
             runtime_hooks=['rthook_onnxruntime.py'],
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

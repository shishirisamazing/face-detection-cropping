# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import importlib.util as _importlib_util
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, collect_dynamic_libs, copy_metadata

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
    _skip = {'onnxruntime', 'rembg', 'torch', 'sklearn', 'skimage'}
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

metadata_datas = []
for _metadata_pkg in ('pymatting', 'rembg', 'onnxruntime'):
    try:
        metadata_datas += copy_metadata(_metadata_pkg)
    except Exception:
        pass

_u2net_model_name = 'isnet-general-use.onnx'
_u2net_model_source = os.path.join(os.path.expanduser('~'), '.u2net', _u2net_model_name)
_u2net_datas = []
if os.path.exists(_u2net_model_source):
    _u2net_datas.append((_u2net_model_source, '.u2net'))
else:
    print("WARNING: {} not found at {}. rembg may require runtime download.".format(
        _u2net_model_name, _u2net_model_source
    ))

# collect_submodules('rembg') often fails to discover session submodules because
# importing rembg triggers onnxruntime which crashes on Windows CI.  Explicitly
# list the core rembg modules so they are always bundled.  new_session() relies
# on BaseSession.__subclasses__(), which only works if the session modules have
# actually been imported.
_rembg_explicit = [
    'rembg.bg',
    'rembg.session_factory',
    'rembg.sessions',
    'rembg.sessions.base',
    'rembg.sessions.dis_general_use',
    'rembg.sessions.dis_anime',
    'rembg.sessions.u2net',
    'rembg.sessions.u2netp',
    'rembg.sessions.u2net_human_seg',
    'rembg.sessions.u2net_cloth_seg',
    'rembg.sessions.silueta',
]

# rembg renamed some session modules across versions; include whichever exists.
for _maybe_session in ('rembg.sessions.isnet_general_use', 'rembg.sessions.isnet_anime'):
    if _importlib_util.find_spec(_maybe_session) is not None:
        _rembg_explicit.append(_maybe_session)

# Collect scikit-image (skimage) — required by rembg.bg for morphological ops.
# Use targeted collection instead of collect_all to avoid pulling in torch
# (skimage has optional torch compatibility that causes DLL access violations).
skimage_datas = collect_data_files('skimage')
skimage_hiddenimports = collect_submodules('skimage')
skimage_binaries = collect_dynamic_libs('skimage')

all_datas = mp_datas + rembg_datas + onnx_datas + skimage_datas + metadata_datas
all_binaries = mp_binaries + onnx_binaries + skimage_binaries
all_hiddenimports = (mp_hiddenimports + rembg_hiddenimports + onnx_hiddenimports
                     + skimage_hiddenimports + _rembg_explicit
                     + ['filetype', 'pooch', 'pymatting', 'scipy',
                        'tqdm', 'jsonschema'])

if sys.platform == 'win32':
    _vc_runtime_names = {
        'msvcp140.dll',
        'msvcp140_1.dll',
        'vcruntime140.dll',
        'vcruntime140_1.dll',
    }

    # Remove bundled VC runtime DLLs collected from various packages so we can
    # inject a single consistent runtime version from System32.
    all_binaries = [
        _bin for _bin in all_binaries
        if os.path.basename(_bin[0]).lower() not in _vc_runtime_names
    ]

    _system32 = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'System32')
    _qt_bin_dest = os.path.join('PyQt5', 'Qt5', 'bin')
    for _name in sorted(_vc_runtime_names):
        _src = os.path.join(_system32, _name)
        if os.path.exists(_src):
            # Keep copies in app root and Qt bin for stable DLL resolution.
            all_binaries.append((_src, '.'))
            all_binaries.append((_src, _qt_bin_dest))
        else:
            print("WARNING: {} not found in {}".format(_name, _system32))

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
             ] + _u2net_datas + all_datas,
             hiddenimports=all_hiddenimports,
             hookspath=[],
             runtime_hooks=['rthook_onnxruntime.py'],
             excludes=['onnx.reference', 'torch', 'torchvision', 'torchaudio',
                       'tensorflow', 'tensorboard'],
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

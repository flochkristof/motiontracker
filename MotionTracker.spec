# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for MotionTracker
# Works on Windows, macOS, and Linux

import sys
from pathlib import Path

block_cipher = None

# Paths
src_dir = Path('src/MotionTrackerBeta')
images_dir = src_dir / 'images'

# Collect all data files (images and styles)
style_dir = src_dir / 'style'

datas = [
    (str(images_dir), 'MotionTrackerBeta/images'),
    (str(style_dir), 'MotionTrackerBeta/style'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'cv2',
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.interpolate',
    'scipy.optimize',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'pandas',
    'openpyxl',
    'cvxopt',
    'cvxpy',
    'pynumdiff',
    'tqdm',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.sip',
    'scipy._lib.array_api_compat',
    'scipy._lib.array_api_compat.numpy',
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy._lib.array_api_compat.numpy.linalg',
    'scipy.special',
    'scipy.special._support_alternative_backends',
    'scipy.integrate',
    'scipy.integrate._quadrature',
    'osqp',
    'osqp.builtin',
    'osqp.ext_builtin',
    'osqp._version',
    'osqp.interface',
]

a = Analysis(
    ['src/MotionTrackerBeta/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platform-specific settings
if sys.platform == 'darwin':
    # macOS: Create .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='MotionTracker',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=True,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='src/MotionTrackerBeta/images/logo.ico',  # Will be ignored, need .icns for Mac
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='MotionTracker',
    )
    app = BUNDLE(
        coll,
        name='MotionTracker.app',
        icon=None,  # Add 'src/MotionTrackerBeta/images/logo.icns' if you create one
        bundle_identifier='com.motiontracker.app',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': '0.1.7',
        },
    )
else:
    # Windows and Linux: Single executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='MotionTracker',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,  # No console window (GUI app)
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='src/MotionTrackerBeta/images/logo.ico',
    )

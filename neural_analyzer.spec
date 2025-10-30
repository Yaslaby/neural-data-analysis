# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('dialogs.py', '.'),
        ('PlotManager.py', '.'),
        ('annotation.py', '.'),
        ('signal_processing.py', '.'),
        ('integrated_mne_processing.py', '.'),
        ('MultiChannelLoader.py', '.'),
    ],
    hiddenimports=[
        'scipy.special.cython_special',
        'scipy._lib.messagestream',
        'sklearn.utils._weight_vector',
        'mne',
        'mne.io',
        'mne.io.base',
        'mne.preprocessing',
        'mne.filter',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'pyqtgraph',
        'numpy',
    ],
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuralDataAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for release to hide console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NeuralDataAnalyzer',
)

app = BUNDLE(
    coll,
    name='NeuralDataAnalyzer.app',
    icon=None,  # Add icon path here if you have one: 'icon.icns'
    bundle_identifier='com.neuraldata.analyzer',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
    },
)
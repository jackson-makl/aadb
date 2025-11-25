# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('app.py', '.'), ('data_loader.py', '.'), ('risk_predictor.py', '.'), ('config.py', '.'), ('models', 'models'), ('data', 'data'), ('.streamlit', '.streamlit')]
binaries = []
hiddenimports = ['sklearn.utils._weight_vector', 'sklearn.neighbors._partition_nodes', 'sklearn.ensemble._forest', 'sklearn.ensemble', 'sklearn.tree._tree', 'sklearn.preprocessing._label', 'streamlit', 'altair', 'pandas', 'numpy', 'matplotlib', 'seaborn']
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('altair')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app_launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AADB_Risk_Predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AADB_Risk_Predictor',
)
app = BUNDLE(
    coll,
    name='AADB_Risk_Predictor.app',
    icon=None,
    bundle_identifier='org.aimahead.aadb',
)

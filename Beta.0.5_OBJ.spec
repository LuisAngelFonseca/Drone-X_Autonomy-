# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['Beta.0.5_OBJ.py'],
             pathex=['C:\\Users\\Kamil\\Documents\\Semestre_i\\.DroneX\\Github_Repo\\Drone-X_Autonomy-'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.datas += [('Resources\\logo.png','C:\\Users\\Kamil\\Documents\\Semestre_i\\.DroneX\\Github_Repo\\Drone-X_Autonomy-\\Resources\\logo.png', 'DATA')]
a.datas += [('mask_values.pkl','C:\\Users\\Kamil\\Documents\\Semestre_i\\.DroneX\\Github_Repo\\Drone-X_Autonomy-\\mask_values.pkl', 'DATA')]
a.datas += [('opencv_videoio_ffmpeg420_64.dll','C:\\Python38\\Lib\\site-packages\\cv2\\opencv_videoio_ffmpeg420_64.dll', 'DATA')]
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Beta.0.5_OBJ',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )

from setuptools import setup
from distutils.sysconfig import get_python_lib
import os

setup(name='ccaction',
      version='0.17.0',
      description='Cybercore - Human action recognition development project',
      url='https://github.com/cybercore-co-ltd/ccaction.git',
      author='cybercoreAI',
      author_email='chuong.nguyen@cybercore.co.jp',
      packages=['ccaction'],
      zip_safe=False)

print("Your Site Packages path:")
lib_path = get_python_lib()
print(lib_path)
for mmpackage in ['mmdet', 'mmpose', 'mmaction']:
    cfg_path = f'configs/{mmpackage}'
    if os.path.islink(cfg_path):
        print(f'unlink {cfg_path}')
        os.unlink(cfg_path)
    mmcfg_path = os.path.join(lib_path, mmpackage, '.mim/configs')
    print(mmcfg_path)
    if os.path.isdir(mmcfg_path):
        print(f"adding {cfg_path}")
        os.symlink(mmcfg_path, cfg_path)

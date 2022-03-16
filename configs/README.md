# Create Config folder

Since action recognition depends on several steps/packages, such as `mmdetection`, `mmpose`, `mmtrack` and `mmaction`, to avoid the redundant and messy files, we create each Soft-link folder for each package configs. 
When running `python setup.py develop`, the folder config should be created automatically, asin the following snippet code

```shell
print("Your Site Packages path:")
lib_path = get_python_lib()
print(lib_path)
for mmpackage in ['mmdet', 'mmpose', 'mmaction']:
    cfg_path = f'configs/{mmpackage}'
    if os.path.islink(cfg_path):
        print(f'unlink {cfg_path}')
        os.unlink(cfg_path)

    print(f"adding {cfg_path}")
    os.symlink(os.path.join(lib_path, mmpackage, '.mim/configs'), cfg_path)
```

Hence, the configs folder structure should look like this:
```
configs
    |-mmaction/
    |-mmdet/
    |-mmpose/
    |-{cc-developed-cfg}
```

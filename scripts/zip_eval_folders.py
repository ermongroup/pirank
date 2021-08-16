import glob
import os
experiment_folders = './tmp/*/eval'
zipfile = 'models_65.zip'

for folder in glob.glob(experiment_folders):
    path = folder.rsplit('/',2)
    os.system(f'cp -a {folder} {path[0]}/{"_".join(path[1:])}')

path = experiment_folders.rsplit('/',1)
os.system(f'zip -r {zipfile} {"_".join(path)}')


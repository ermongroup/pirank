import glob
experiment_folders = './tmp/*/eval'
zipfile = 'models_65.zip'

for folder in glob.glob(experiment_folders):
    path = folder.rsplit('/',2)
    print(f'cp -a {folder} {path[0]}/{"_".join(path[1:])}')

path = experiment_folders.rsplit('/',1)
print(f'zip -r {zipfile} {"_".join(path)}')


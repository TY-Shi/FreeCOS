import os
"""
Make STARE list
"""

test_image = '/mnt/nas/sty/codes/Unsupervised/Data/STARE/test/img/'
files = os.listdir(test_image)
print("files", files)
print("lenfiles", len(files))

meta_file = '/mnt/nas/sty/codes/Unsupervised/Data/STARE/split/test_img.txt'
train_file = open(meta_file, 'a')
for name in files:
    #spilt_name = name.split('_')[0]
    spilt_name = name
    print("spilt：",spilt_name)
    train_file.write(str(spilt_name))
    train_file.write('\n')



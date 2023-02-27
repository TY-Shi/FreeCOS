import os
"""
Make STARE list
"""

#test_image = '/mnt/nas/sty/codes/Unsupervised/Data/STARE/test/img/'
test_image = "./train/fake_onlythin_gt/"
files = os.listdir(test_image)
print("files", files)
print("lenfiles", len(files))

meta_file = './split/train_fakevessel_only.txt'
train_file = open(meta_file, 'a')
for name in files:
    #spilt_name = name.split('_')[0]
    spilt_name = name
    print("spilt：",spilt_name)
    train_file.write(str(spilt_name))
    train_file.write('\n')



import os
"""
Make crack200 list
"""
# train_image = 'F://Datasets//Curvilinear_data//cracktree200//cracktree200rgb//train_image//'
# files = os.listdir(train_image)
# print("files",files)
# print("lenfiles",len(files))
#
# meta_file = 'F://Datasets//Curvilinear_data//cracktree200//cracktree200rgb//train.txt'
# train_file = open(meta_file, 'a')
# for name in files:
#     spilt_name = name.split('.')[0]
#     train_file.write(str(spilt_name))
#     train_file.write('\n')
#
#
# test_image = 'F://Datasets//Curvilinear_data//cracktree200//cracktree200rgb//test_image//'
# files = os.listdir(test_image)
# print("files",files)
# print("lenfiles",len(files))
#
# meta_file = 'F://Datasets//Curvilinear_data//cracktree200//cracktree200rgb//test.txt'
# train_file = open(meta_file, 'a')
# for name in files:
#     spilt_name = name.split('.')[0]
#     train_file.write(str(spilt_name))
#     train_file.write('\n')

"""
DSA_XH
"""

# test_image = 'F://Datasets//Curvilinear_data//DSA-Privacy Preserving Data//DSA_image//'
# files = os.listdir(test_image)
# print("files", files)
# print("lenfiles", len(files))
#
# meta_file = 'F://Datasets//Curvilinear_data//DSA-Privacy Preserving Data//test.txt'
# train_file = open(meta_file, 'a')
# for name in files:
#     spilt_name = name.split('.')[0]
#     train_file.write(str(spilt_name))
#     train_file.write('\n')


"""
DSA_Select
"""

#test_image = './fake_grayvessel/'
#test_image = './train/img/'
# test_image = './test/img/'
# files = os.listdir(test_image)
# print("files", files)
# print("lenfiles", len(files))
#
# meta_file = './split/test_img.txt'
# train_file = open(meta_file, 'a')
# for name in files:
#     #spilt_name = name.split('_')[0]
#     spilt_name = name
#     print("spilt：",spilt_name)
#     train_file.write(str(spilt_name))
#     train_file.write('\n')
#
test_image = '/mnt/nas/sty/codes/Unsupervised/Data/DRIVE/train/fake_vessel/'
files = os.listdir(test_image)
print("files", files)
print("lenfiles", len(files))

meta_file = '/mnt/nas/sty/codes/Unsupervised/Data/DRIVE/split/train_fakevessel.txt'
train_file = open(meta_file, 'a')
for name in files:
    #spilt_name = name.split('_')[0]
    spilt_name = name
    print("spilt：",spilt_name)
    train_file.write(str(spilt_name))
    train_file.write('\n')



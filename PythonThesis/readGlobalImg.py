# # import cv2
# # import glob
# #
# # imdir = 'image/'
# # ext = ['png', 'jpg']    # Add image formats here
# #
# # files = []
# # [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
# #
# # images = [cv2.imread(file) for file in files]
# #
# # for i in images:
# #     cv2.imshow("image",i)
# #
# # cv2.waitKey(0)
#
# import sys
# import glob
# import os.path
#
# list_of_files = glob.glob('image/*.jpg') #500 files
#
# for file_name in list_of_files:
#     print(file_name)
#
#     # This needs to be done *inside the loop*
#     f= open(file_name, 'r')
#     lst = []
#     for line in f:
#        line.strip()
#        line = line.replace("\n" ,'')
#        line = line.replace("//" , '')
#        lst.append(line)
#     f.close()
#
#     f=open(os.path.join('image/',
#     os.path.basename(file_name)) , 'w')
#
#     for line in lst:
#        f.write(line)
#     f.close()

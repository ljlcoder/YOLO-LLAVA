import glob
#win下写法，linux同学自己改下分隔符
all_file_name_list = glob.glob('D:\\PycharmProject\\yolov7-main\\NEU-DET\\valid\\images\\*.*')

with open('D:\\PycharmProject\\yolov7-main\\NEU-DET\\val.txt','w') as f:
    for file_name in all_file_name_list:
        file_name = file_name.replace("\\", "\\\\")
        f.write(file_name)
        f.write('\n')

all_file_name_list = glob.glob('D:\\PycharmProject\\yolov7-main\\NEU-DET\\train\\images\\*.*')

with open('D:\\PycharmProject\\yolov7-main\\NEU-DET\\train.txt','w') as f:
    for file_name in all_file_name_list:
        file_name = file_name.replace("\\", "\\\\")
        f.write(file_name)
        f.write('\n')


import os
from os import getcwd

classes = ['ganyu','keqing']
sets=['train']

if __name__ == '__main__':
    wd = getcwd()
    for se in sets:
        list_file = open('cls_'+ se + '.txt','w')

        dataset_path = se
        types_name = os.listdir(dataset_path)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)
            photos_path = os.path.join(dataset_path, type_name)
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _,postfix = os.path.splitext(photo_name)#分类文件名和扩展名
                if postfix not in ['.jpg','.png','.jpeg']:
                    continue
                list_file.write(str(cls_id)+';'+ '%s'%os.path.join(wd, os.path.join(photos_path,photo_name)))
                list_file.write('\n')
        list_file.close()
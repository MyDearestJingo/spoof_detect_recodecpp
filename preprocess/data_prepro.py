import os
import cv2
from tools.detect import MtcnnDetector

'''
    File Structure
    ../preprocess
        /data_prepro.py
        /tools
            /detect.py
            ...
        /preproc_data
            /re_train_neg_align
                /test_part
                /train_part
            /re_train_pos_align
                /test_part
                /train_part
        /org_data
            /re_train_neg_align
                /test_part
                /train_part
            /re_train_pos_align
                /test_part
                /train_part
        /nets
            ...
'''

in_path = "org_data/"
out_path = "preproc_data/"

data_path_list = [
    # "re_train_neg_align/test_part/",
    # "re_train_neg_align/train_part/",
    "re_train_pos_align/test_part/"
    # "re_train_pos_align/train_part/"
]

def crop_images(img, bboxs): # 【MD】bboxs:人脸区域
    num_face = bboxs.shape[0]
    h, w = img.shape[:2]
    croped_bboxs = []
    cropped = []
    for i in range(num_face):
        b_w = bboxs[i, 2] - bboxs[i, 0]
        b_h = bboxs[i, 3] - bboxs[i, 1]
        l_x = max(bboxs[i, 0] - 0.4 * b_w, 0)
        l_y = max(bboxs[i, 1] - 0.4 * b_h, 0)
        r_x = min(bboxs[i, 2] + 0.4 * b_w, w)
        # r_y = min(bboxs[i, 3] + 0.4 * b_h, h)
        r_y = bboxs[i, 3]
        cb_0 = (0.4 * b_w) if bboxs[i, 0] > (0.4 * b_w)  else bboxs[i, 0]           #xL
        cb_1 = (0.4 * b_h) if bboxs[i, 1] > (0.4 * b_h)  else bboxs[i, 1]           #yL
        cb_2 = (1.4 * b_w) if bboxs[i, 0] > (0.4 * b_w)  else bboxs[i, 0] + b_w     #xR
        cb_3 = cb_2 + b_h                                                           #yR
        cropped_img = img[int(l_y):int(r_y), int(l_x):int(r_x)]
        c_h, c_w = cropped_img.shape[:2]
        cropped.append(cropped_img)
        croped_bboxs.append(int(cb_0))
        croped_bboxs.append(int(cb_1))
        croped_bboxs.append(int(min(c_w-cb_0,b_w)))
        croped_bboxs.append(int(min(c_h-cb_1,b_h)))
    return cropped , croped_bboxs

if __name__ == '__main__':
    mtcnn_detector = MtcnnDetector(min_face_size=24, use_cuda=False)
    for i_path in range(len(data_path_list)):
        data_path = data_path_list[i_path]
        img_name_list = os.listdir(in_path+data_path)
        num_imgs = len(img_name_list)
        cropped_img_list = list()
        cropped_bboxs_list = list()
        img_list = list()
        bboxs_list = list()
        print("Now at Part "+str(i_path+1)+" of "+str(len(data_path_list))+" "+ data_path + " | number of imgs: "+str(num_imgs))
        
        for i in range(num_imgs):
            img_name = img_name_list[i]
            print(">>> Processing: "+str(i)+" of "+str(num_imgs)+" "+data_path+img_name, end='\r')
            img = cv2.imread(in_path+data_path+img_name)
            img_list.append(img)
            bbox, _ = mtcnn_detector.detect_face(img)
            bboxs_list.append(bbox)
            cropped_img, cropped_bboxs = crop_images(img, bbox)
            cropped_img_list.append(cropped_img)
            cropped_bboxs_list.append(cropped_bboxs)
        print("                                                                                       ")
        print("=== Process "+data_path+" complete")
        # output
#         os.mknod("")
        bboxs_file = open(out_path+data_path+"bboxs_list.txt", 'w')
        cp_bboxs_file = open(out_path+data_path+"cropped_bboxs_list.txt", 'w')
        
        for i in range(num_imgs): # for every img
            # save bboxs for every img
            bboxs_file.write(img_name_list[i]+ ' '+ str(bboxs_list[i].shape[0])+ ' ')
            
            for b in range(bboxs_list[i].shape[0]): # number of faces in img[i]
                img_file_name = str(b)+'_'+img_name_list[i]    
                cv2.imwrite(out_path+data_path+img_file_name,cropped_img_list[i][b])
                for n in range(bboxs_list[i].shape[1]): # number of coordinates of face[n]
                    bboxs_file.write(str(bboxs_list[i][b, n])+' ')
            bboxs_file.write('\n')
            # save cropped bbox coordinates for every img
            cp_bboxs_file.write(img_name_list[i] + ' ' + str(bboxs_list[i].shape[0]) + ' ')
            for position in cropped_bboxs_list[i]:
                cp_bboxs_file.write(str(position) + ' ')
            cp_bboxs_file.write('\n')




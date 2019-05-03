#include "include/svm_tools.h"
using namespace std;

/*
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
*/
/*
    Args: 
        FILE: the fstream object that opens the cropped_bboxs.txt of the type(neg/pos, for train/test)
            of data to read.
        dir: the dir of this type(neg/pos, for train/test) of images
        v_data: the output which is a vector of struct data containing the Mat object with its
            cropped_bbox.
*/
void read_data(fstream &FILE, string dir, vector<data> &v_data){
    while(!FILE.eof()){
        string org_filename;
        FILE>>org_filename;
        int num_face = 0;
        FILE>>num_face;
        for(int i=0;i<num_face;i++){
            data new_data;
            string path = dir + to_string(i) + "_" + org_filename;
            new_data.cropped_img = imread(path);
            for(int j=0;j<4;j++){
                FILE>>new_data.cropped_bbox[j];
            }
            v_data.pop_back(new_data);
        }
    }
}
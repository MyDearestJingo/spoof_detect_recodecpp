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
void read_data(ifstream &FILE, string dir, vector<data> &v_data){
    int count = 0;
    while(!FILE.eof()){
        #ifdef DEBUG
        count++;
        cout<<"Reading No."<<count<<endl;
        #endif
        string org_filename;
        FILE>>org_filename;
        int num_face = 0;
        FILE>>num_face;
        for(int i=0;i<num_face;i++){
            data new_data;
            string path = dir + to_string(i) + "_" + org_filename;
            Mat src = imread(path);
            cvtColor(src, new_data.cropped_img, COLOR_BGR2GRAY);
            for(int j=0;j<2;j++){
                FILE>>new_data.cropped_bbox[j];
            }
            int right_down_x, right_down_y;
            FILE>>right_down_x;
            FILE>>right_down_y;
            new_data.cropped_bbox[2] = right_down_x - cropped_bbox[0];
            new_data.cropped_bbox[3] = right_down_y - cropped_bbox[1];
            v_data.push_back(new_data);
        }
    }
}
/*
    Args:
        v_data: a vector of struct data containing the Mat object with its cropped_bbox.
        fd: output, a float[n_samples][256*2+59] array for feature vector, where 256 is the length of 
            face_hist and nonface_hist and 59 is the length of lbp_hist 
*/
void preproc_data(vector<data> &v_data,const int n_samples, float fd[][256*2+59]){
    for(int i=0;i<n_samples;i++){
        #ifdef DEBUG
        cout<<"preproc_data No."<<i<<endl;
        #endif // DEBUG
        MatND face_hist, nonface_hist;
        #ifdef DEBUG
        cout<<"Enter context feature"<<endl;
        #endif // DEBUG
        get_context_feature(v_data[i].cropped_img, v_data[i].cropped_bbox, face_hist, nonface_hist);
        MatND lbp; // 无用，日后可以删掉省点内存
        MatND lbp_hist;
        #ifdef DEBUG
        cout<<"Enter lbp feature"<<endl;
        #endif // DEBUG
        get_lbp_feature(v_data[i].cropped_img, RADIUIS, NEIGHBORS, lbp, lbp_hist);
        for(int j=0;j<256;j++){ // 256 is the length of face_hist and nonface_hist
            fd[i][j] = face_hist.at<float>(j,0);
            fd[i][j+256] = nonface_hist.at<float>(j,0);
        }
        for(int j=0;j<59;j++){ // 59 is the length of lbp_hist
            fd[i][j+256*2] = lbp_hist.at<float>(j,0);
        }
    }

}

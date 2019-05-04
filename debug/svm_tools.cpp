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
    cout<<"Loading dir: "<<dir<<endl;
    while(!FILE.eof()){

        string org_filename;
        FILE>>org_filename;
        #ifdef DEBUG
        count++;
        cout<<"Reading No."<<count<<" Filename: "<<org_filename<<endl;
        #endif
        int num_face = 0;
        FILE>>num_face;
        for(int i=0;i<num_face;i++){
            data new_data;
            string path = dir + to_string(i) + "_" + org_filename;
            Mat src = imread(path);
            cvtColor(src, new_data.cropped_img, COLOR_BGR2GRAY);
            for(int j=0;j<4;j++){
                FILE>>new_data.cropped_bbox[j];
            }
            // int right_down_x, right_down_y;
            // FILE>>right_down_x;
            // FILE>>right_down_y;
            // new_data.cropped_bbox[2] = right_down_x - cropped_bbox[0];
            // new_data.cropped_bbox[3] = right_down_y - cropped_bbox[1];
            v_data.push_back(new_data);
        }
    }
}
/*
    Args:
        v_data: a vector of struct data containing the Mat object with its cropped_bbox.
        fd: output, a float[n_samples][1+256*2+59] array for feature vector, where 1 is the class flag of the sample,
            256 is the length of face_hist and nonface_hist and 59 is the length of lbp_hist 
*/
void preproc_data(vector<data> &v_data,const int cls_flag, const int n_samples, float fd[][FEATURE_DIM]){
    for(int i=0;i<n_samples;i++){
        #ifdef DEBUG
        cout<<"preproc_data No."<<i<<'\r';
        #endif // DEBUG
        MatND face_hist, nonface_hist;
        cout<<"Generate Feature Vector "<<i+1<<" of "<<n_samples<<" Samples"<<'\r';
        get_context_feature(v_data[i].cropped_img, v_data[i].cropped_bbox, face_hist, nonface_hist);
        MatND lbp; // 无用，日后可以删掉省点内存
        MatND lbp_hist;
        get_lbp_feature(v_data[i].cropped_img, RADIUIS, NEIGHBORS, lbp, lbp_hist);
        fd[i][0] = (float)cls_flag;
        for(int j=1;j<256+1;j++){ // 256 is the length of face_hist and nonface_hist
            fd[i][j] = face_hist.at<float>(j,0);
            fd[i][j+256] = nonface_hist.at<float>(j,0);
        }
        for(int j=0;j<59;j++){ // 59 is the length of lbp_hist
            fd[i][j+256*2+1] = lbp_hist.at<float>(j,0);
        }
    }
    cout<<endl;
}


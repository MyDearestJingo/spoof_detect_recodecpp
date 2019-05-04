// Test svm_tools.cpp
#include "include/svm_tools.h"

int main(){
    // vector<data> v_data;
    // string dir = 
    //     "/home/mdlu/spoof_detect_recodecpp/preprocess/preproc_data/re_train_pos_align/test_part/";
    // string cropped_bboxs_filename = "cropped_bboxs_list.txt";
    // ifstream FILE;
    // FILE.open(dir+cropped_bboxs_filename);
    // read_data(FILE, dir, v_data);

    // #ifdef DEBUG
    // for(int i=0;i<0;i++){
    //     imshow("preview",v_data[i].cropped_img);
    //     for(int j=0;j<4;j++){
    //         cout<<v_data[i].cropped_bbox[j]<<' ';
    //     }
    //     cout<<endl;
    //     waitKey(0);
    // }
    // #endif

    // float fd[2500][1+256*2+59];
    // preproc_data(v_data, v_data.size(), fd);

    string dir_list[4] = {
        "/home/mdlu/spoof_detect_recodecpp/preprocess/preproc_data/re_train_neg_align/test_part/",
        "/home/mdlu/spoof_detect_recodecpp/preprocess/preproc_data/re_train_neg_align/train_part/",
        "/home/mdlu/spoof_detect_recodecpp/preprocess/preproc_data/re_train_pos_align/test_part/",
        "/home/mdlu/spoof_detect_recodecpp/preprocess/preproc_data/re_train_pos_align/train_part/"
    };
    // const int dim_features = 1+256*2+59;
    string cropped_bboxs_filename = "cropped_bboxs_list.txt";
    ifstream FILE;
    ofstream FD_OUT;
    vector<data> v_data;    
    float fd[3000][FEATURE_DIM];
    for(int i=0;i<4;i++){
        FILE.open(dir_list[i]+cropped_bboxs_filename);
        read_data(FILE, dir_list[i], v_data);
        int n_samples = v_data.size();
        int cls_flag = (i<2)? -1 : 1;
        preproc_data(v_data,cls_flag,n_samples, fd);
        FD_OUT.open(dir_list[i]+"fd_out.txt");
        for(int j=0;j<n_samples;j++){
            for(int k=0;k<FEATURE_DIM;k++){
                FD_OUT<<fd[j][k]<<' ';
            }
            FD_OUT<<endl;
        }
        v_data.clear();
        FILE.close();
        FD_OUT.close();
    }
    return 0;
}
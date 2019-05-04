// Test svm_tools.cpp
#include "include/svm_tools.h"

int main(){
    vector<data> v_data;
    string dir = 
        "/home/mdlu/spoof_detect_recodecpp/debug/preproc_data/re_train_neg_align/test_part/";
    string cropped_bboxs_filename = "cropped_bboxs_list.txt";
    ifstream FILE;
    FILE.open(dir+cropped_bboxs_filename);
    read_data(FILE, dir, v_data);

    #ifdef DEBUG
    for(int i=0;i<0;i++){
        imshow("preview",v_data[i].cropped_img);
        for(int j=0;j<4;j++){
            cout<<v_data[i].cropped_bbox[j]<<' ';
        }
        cout<<endl;
        waitKey(0);
    }
    #endif

    float fd[10][256*2+59];
    preproc_data(v_data, 10, fd);

    return 0;
}
#include "./include/features.h"


int main(int argc, char** argv){
    string img_path = "/home/mdlu/spoof_detect_recodecpp/preprocess/preproc_data/re_train_pos_align/test_part/0_1554818939.8856304.jpg";
    cout<<"test start"<<endl;
    Mat src, dst;
    src = imread(img_path);
    if(!src.data){
        cout<<"Error: Img read failed!"<<endl;
        return -1;
    }
    // imshow("original",src);

    // float face_hist[256] = {0.0};
    // float nonface_hist[256] = {0.0};
    cvtColor(src, dst, COLOR_BGR2GRAY);
    imshow("gray", dst);
    waitKey(0);
    // cropped_bbox[4] = {xL,yL,w,h};
    int cropped_bbox[4] = {51, 60, 129, 152 }; 
    if(spoof_detect(dst, cropped_bbox)) cout<<"Complete"<<endl;
    return 0;
}

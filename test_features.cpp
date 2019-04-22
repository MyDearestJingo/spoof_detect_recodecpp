#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "features.h"
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;

string img_path = "img.jpg";

int main(int argc, char** argv){
    Mat src, dst;
    src = imread(argv[1], 1);
    if(!src.data){
        cout<<"Error: Img read failed!"<<endl;
        return -1;
    }
    float face_hist[256] = {0.0};
    float nonface_hist[256] = {0.0};
    int crop_bboxs[4] = {10,10,100,100}; // for test
    get_context_feature(src,crop_bboxs, face_hist, nonface_hist);
}
#pragma once 
#include <opencv2/opencv.hpp>
using namespace cv;

void get_context_feature(
    // input arg
    const Mat& img, 
    const int* crop_bbox,
    // output
    MatND& face_hist,
    MatND& nonface_hist)；
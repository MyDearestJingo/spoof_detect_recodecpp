#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <bitset>

// #define DEBUG
#define RADIUIS 2
#define NEIGHBORS 8

using namespace cv;
using namespace std;

void get_context_feature(const Mat& img, const int* crop_bbox, MatND& face_hist, MatND& nonface_hist);

template <typename _tp>
void getUniformPatternLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors);
int getHopTimes(int n);
void get_lbp_feature(const Mat& src, const int radius, const int neighbors, MatND &lbp, MatND &lbp_hist);

bool spoof_detect(Mat &img, int *cropped_bbox);
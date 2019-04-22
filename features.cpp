#include <opencv.h>
#include <iostream>

using namespace std;

/*  Log
    Apr 22nd:
        - 完成第一次编写，可优化点在两个循环的位置，具体需要查询ARM处理器的指令集
        - calcHist()与【裁剪图像】尚未进行验证运行
*/
void get_context_feature(
    // input arg
    const Mat* img, 
    const int* crop_bbox,
    // output
    float* face_hist,
    float* nonface_hsit)
    {
    // 使用opencv绘制图像直方图
    // 直方图属性：
    float img_hist[256] = {0.0}; // full img hist array
    float face_hist[256] = {0.0}; // face region hist array
    float nonface_hist[256] = {0.0}; // non-face region hist array
    int histSize = 256;
    int channles = 0;
    int ranges[2] = {0, 256};
    // void calcHist(images, nimages, channels, mask, hist, dims, histSize, ranges, uniform, accumulate)
    // dims: 直方图描述的维度，若只针对灰度图，则只是1-dim，对RGB三通道都进行计算则为3-dim
    // uniform：默认为true，是否需要改为false
    // 计算全局直方图
    calcHist(img, 1, &channels, NULL, img_hist, dims=1, &histSize, ranges);
    // 根据crop_bbox坐标裁剪图像
    Rect roi(crop_bbox[0],crop_bbox[1],crop_bbox[2],crop_bbox[3]);
    Mat face = img(roi);
    // 计算人脸区域直方图
    calcHist(face, 1, &channels, NULL, img_hist, dims=1, &histSize, ranges);
    float sum = 0.0;
    // 非人脸区域直方图 = 全图直方图 - 人脸区域直方图 || 人脸区域归一化前sum的计算
    for(int i=0;i<256;i++){
        nonface_hist[i] = img_hist[i] - face_hist[i];
        sum += face_hist[i];
    }
    // 归一化
    for(int i=0;i<256;i++){
        face_hist[i] = face_hist[i]/sum;
    }

}
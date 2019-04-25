#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
//#include "features.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;

string img_path = "lena.jpeg";

void get_context_feature(
    // input arg
    const Mat& img, 
    const int* crop_bbox,
    // output
    MatND& face_hist,
    MatND& nonface_hist)
    {
    cout<<"get in func"<<endl;
    // 使用opencv绘制图像直方图
    // 直方图属性：
    // hbins: 色调等级,色调为0则为黑色，此处设置为256个级别; 
    // sbins: 饱和度等级,0时表示灰度图像;
    int hbins = 256, sbins = 1; 
    int histSize[] = {hbins,sbins};
    int channles[] = {0};
    // 设置色调的取之范围为{0,256}
    float hranges[] = { 0, 256 };
    // 由于使用灰度图作为输入，故饱和度取0即可
    float sranges[] = { 0, 1 };
    const float* ranges[] = { hranges, sranges };
    // void calcHist(images, nimages, channels, mask, hist, dims, histSize, ranges, uniform, accumulate)
    // dims: 直方图描述的维度，若只针对灰度图，则只是1-dim，对RGB三通道都进行计算则为3-dim
    // uniform：默认为true，是否需要改为false
    // 计算全局直方图
    MatND img_hist;
    cout<<"clac img_hist ..."<<endl;
    calcHist(&img, 1, channles, Mat(), img_hist, 1, histSize, ranges);
    // 根据crop_bbox坐标裁剪图像
    Rect roi(crop_bbox[0],crop_bbox[1],crop_bbox[2],crop_bbox[3]);
    Mat face = img(roi);

    cout<<"("<<face.rows<<","<<face.cols<<")"<<endl;
    // imshow(" ",face);
    // waitKey(0);
    // 输出face像素值
    int x = 50;
    int y = 50;
    cout<<"face "<<x<<","<<y<<" :"<<face.at<float>(x,y)<<endl;
    // 计算人脸区域直方图
    cout<<"clac face_hist ..."<<endl;
    calcHist(&face, 1, channles, Mat(), face_hist, 1, histSize, ranges);
    // calcHist(&face, 1, channles, Mat(), nonface_hist, 1, histSize, ranges);
    
    // 预输出face_hist未归一化版本
    for(int i=40;i<50;i++){
        cout<<i<<" : "<<face_hist.at<float>(i,0)<<endl;
    }
    // 预输出img_hist
    for(int i=0;i<hranges[1];i++){
        // cout<<i+1<<" : "<<img_hist.at<float>(i,0)<<endl;
    }

    // 初始化nonface_hist
    calcHist(&face, 1, channles, Mat(), nonface_hist, 1, histSize, ranges);
    // nonface_hist = face_hist;  // <-很奇怪这一句会影响face_hist的值
    float sum = 0.0;
    // 非人脸区域直方图 = 全图直方图 - 人脸区域直方图 || 人脸区域归一化前sum的计算
        // for(int i=0;i<256;i++){
        //     nonface_hist[i] = img_hist[i] - face_hist[i];
        //     sum += face_hist[i];
        // }
    cout<<"clac nonface_hist ..."<<endl;
    for( int h = 0; h < hbins; h++ ){
        for( int s = 0; s < sbins; s++ ){   
            // cout<<"h = "<<h<<" | s = "<<s<<endl;
            // float binVal = hist.at<float>(h, s);
            // int intensity = cvRound(binVal*255/maxVal);
            // rectangle( histImg, Point(h*scale, s*scale),
            //             Point( (h+1)*scale - 1, (s+1)*scale - 1),
            //             Scalar::all(intensity),
            //             CV_FILLED );
            // cout<<"img_hist: "<<img_hist.at<float>(h,s)<<" | "
                // <<"face_hist: "<<face_hist.at<float>(h,s)<<endl;;
            nonface_hist.at<float>(h,s) = img_hist.at<float>(h,s) - face_hist.at<float>(h,s);
            sum += face_hist.at<float>(h,s);
        }
    }

    cout<<"nonface_hist "<<56<<" : "<<nonface_hist.at<float>(56,0)<<endl;

    // 归一化
    cout<<"Normalize ..."<<endl;
    cout<<"sum = "<<sum<<endl;
    for( int h = 0; h < hbins; h++ ){
        for( int s = 0; s < sbins; s++ ){
            face_hist.at<float>(h,s) /= sum;
        }
    }
    cout<<"func out"<<endl;
    return;
}

int main(int argc, char** argv){
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
    // imshow("gray",dst);
    // waitKey(0);
    MatND face_hist, nonface_hist;
    int crop_bboxs[4] = {10,10,90,90}; // 根据ROI设置，按顺序分别为{起始点x坐标，起始点y坐标，x方向延伸距离，y方向延伸距离}
    get_context_feature(dst, crop_bboxs, face_hist, nonface_hist);

    cout<<"Normalized_face_hist "<<140<<" : "<<face_hist.at<float>(140,0)<<endl;
    // 文件输出以对照结果
    ofstream ofile;
    ofile.open("face_hist.txt",ios::out);
    for(int i=0;i<256;i++){
        ofile<<face_hist.at<float>(i,0)<<endl;
    }
    ofile.close();
    return 0;
}
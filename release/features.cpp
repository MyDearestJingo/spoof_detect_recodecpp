// Version: Release 0.1.0

// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include <opencv2/opencv.hpp>
// #include <iostream>
#include "./include/features.h"

using namespace std;
using namespace cv;

//等价模式LBP特征计算
template <typename _tp>
void getUniformPatternLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors){
    Mat src = _src.getMat();
    //LBP特征图像的行数和列数的计算要准确
    _dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
    Mat dst = _dst.getMat();
    dst.setTo(0);
    //LBP特征值对应图像灰度编码表，直接默认采样点为8位
    uchar temp = 1;
    uchar table[256] = {0};
    for (int i = 0; i < 256; i++)
    {
        if (getHopTimes(i) < 3)
        {
            table[i] = temp;
            temp++;
        }
    }
    //是否进行UniformPattern编码的标志
    bool flag = false;
    //计算LBP特征图
    for (int k = 0; k < neighbors; k++)
    {
        if (k == neighbors - 1)
        {
            flag = true;
        }
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));
        //为双线性插值做准备
        //对采样点偏移量分别进行上下取整
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - x1;
        float ty = ry - y1;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        //循环处理每个像素
        for (int i = radius; i < src.rows - radius; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                //获得中心像素点的灰度值
                _tp center = src.at<_tp>(i, j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = src.at<_tp>(i + x1, j + y1) * w1 + src.at<_tp>(i + x1, j + y2) * w2 + src.at<_tp>(i + x2, j + y1) * w3 + src.at<_tp>(i + x2, j + y2) * w4;
                //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst.at<uchar>(i - radius, j - radius) |= (neighbor > center) << (neighbors - k - 1);
                //进行LBP特征的UniformPattern编码
                if (flag)
                {
                    dst.at<uchar>(i - radius, j - radius) = table[dst.at<uchar>(i - radius, j - radius)];
                }
            }
        }
    }
}
//计算跳变次数
int getHopTimes(int n){
    int count = 0;
    bitset<8> binaryCode = n;
    for (int i = 0; i < 8; i++)
    {
        if (binaryCode[i] != binaryCode[(i + 1) % 8])
        {
            count++;
        }
    }
    return count;
}


void get_context_feature(const Mat& img,const int* crop_bbox, MatND& face_hist, MatND& nonface_hist){
    cout<<"=============enter context_features============="<<endl;

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

    // 计算人脸区域直方图
    cout<<"clac face_hist ..."<<endl;
    calcHist(&face, 1, channles, Mat(), face_hist, 1, histSize, ranges);
    // calcHist(&face, 1, channles, Mat(), nonface_hist, 1, histSize, ranges);

    // 初始化nonface_hist
    nonface_hist = MatND::zeros(hbins,sbins,CV_32FC1);
    // calcHist(&face, 1, channles, Mat(), nonface_hist, 1, histSize, ranges);
    // nonface_hist = face_hist;  // <-很奇怪这一句会影响face_hist的值
    float sum = 0.0;
    // 非人脸区域直方图 = 全图直方图 - 人脸区域直方图 || 人脸区域归一化前sum的计算
    cout<<"clac nonface_hist ..."<<endl;
    for( int h = 0; h < hbins; h++ ){
        for( int s = 0; s < sbins; s++ ){   
            nonface_hist.at<float>(h,s) = img_hist.at<float>(h,s) - face_hist.at<float>(h,s);
            sum += face_hist.at<float>(h,s);
        }
    }

    // 归一化
    cout<<"Normalize ..."<<endl;
    for( int h = 0; h < hbins; h++ ){
        for( int s = 0; s < sbins; s++ ){
            face_hist.at<float>(h,s) /= sum;
        }
    }
    cout<<"=============exit context_features============="<<endl;
    return;
}
void get_lbp_feature(const Mat& src, const int radius, const int neighbors, MatND &lbp, MatND &lbp_hist){
    cout<<"=============enter lbp_features============="<<endl;

    cout<<"calc LBP features ..."<<endl;
    getUniformPatternLBPFeature<float>(src, lbp, radius, neighbors);
    int nbin_x = neighbors*(neighbors-1)+4;
    int nbin_y = 1; 
    int histSize[] = {nbin_x,nbin_y};
    int channles[] = {0};
    // 直方图宽度标定范围
    float axis_z[] = { 0, nbin_x+1 };
    // 只有一个维度
    const float* ranges = { axis_z };
    lbp_hist = Mat::zeros(nbin_x,nbin_y,CV_32FC1);
    cout<<"calc LBP hist ..."<<endl;
    calcHist(&lbp, 1, channles, Mat(), lbp_hist, 1, &nbin_x, &ranges);
    MatND norn_hist;
    normalize(lbp_hist, lbp_hist, 0, 1, NORM_MINMAX,-1,Mat());

    cout<<"=============exit lbp_features============="<<endl;
}

bool spoof_detect(string img_path){
    Mat src, gray;
    src = imread(img_path);
    if(!src.data){
        cout<<"Error: img read failed"<<endl;
        return -1;
    }
    cvtColor(src,gray,COLOR_BGR2GRAY);
    
    MatND face_hist, nonface_hist;
        int crop_bbox[4] = {10,10,90,90};
    get_context_feature(gray, crop_bbox, face_hist, nonface_hist);
        ofstream ofile;
    ofile.open("face_hist.txt",ios::out);
    for(int i=0;i<256;i++){
        ofile<<face_hist.at<float>(i,0)<<endl;
    }
    ofile.close();
    
    MatND lbp;
    MatND lbp_hist;
    int radius = 2;
    int neighbors = 8;
    get_lbp_feature(gray, radius, neighbors, lbp, lbp_hist);

    ofile.open("lbp_hist.txt",ios::out);
    for(int i=1;i<lbp_hist.rows;i++){
        ofile<<lbp_hist.at<float>(i,0)<<endl;
    }
    ofile.close();
    // 文件输出lbp

    return true;
}
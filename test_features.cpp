#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
//#include "features.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <bitset>

using namespace std;
using namespace cv;

string img_path = "lena.jpeg";

// 计算LBP用函数声明
int getHopTimes(int n);
template <typename _tp>
void getUniformPatternLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors);


void get_context_feature(
    // input arg
    const Mat& img, 
    const int* crop_bbox,
    // output
    MatND& face_hist,
    MatND& nonface_hist){
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
        // cout<<i<<" : "<<face_hist.at<float>(i,0)<<endl;
    }
    // 预输出img_hist
    for(int i=0;i<hranges[1];i++){
        // cout<<i+1<<" : "<<img_hist.at<float>(i,0)<<endl;
    }

    // 初始化nonface_hist
    nonface_hist = MatND::zeros(hbins,sbins,CV_32F);
    // calcHist(&face, 1, channles, Mat(), nonface_hist, 1, histSize, ranges);
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
    cout<<"=============exit context_features============="<<endl;
    return;
}
void get_lbp_feature(const Mat& src, const int radius, const int neighbors, MatND &lbp, MatND &lbp_hist){
    cout<<"=============enter lbp_features============="<<endl;
    int binsNumber = neighbors*(neighbors-1)+3;
    int binsRange[] = {0,binsNumber+1};
    // 初始化MatND lbp

    // 计算LBP
    // 由于是对现有算法的验证，则不再写个什么优秀的函数提供调参的功能了
    // 目前参数
    // int neighbors=8;
    // int radius=2;
    /*
    // 下面是自己手撸的LBP代码
    cout<<"Computing LBP ..."<<endl;
    for(int r=radius;r<img.rows-radius;r++){
        for(int c=radius;c<img.cols-radius;c++){
            float sample = img.at<float>(r,c);
            //lbp.at<float>(r-radius,c-radius)
            bitset<8> lbp_val;
            float target = 
                (img.at<float>(r-radius,c) > sample)       // U
                + (img.at<float>(r-1,c+1) > sample)*128       // UR
                + (img.at<float>(r,c+radius) > sample)*64    // R
                + (img.at<float>(r+1,c+1) > sample)*32       // DR
                + (img.at<float>(r+radius,c) > sample)*16   // D
                + (img.at<float>(r+1,c-1) > sample)*8      // DL
                + (img.at<float>(r,c-radius) > sample)*4   // L
                + (img.at<float>(r-1,c-1) > sample)*2;    // UL

            lbp_val[0] = (img.at<float>(r-radius,c) > sample);       // U
            lbp_val[1] = (img.at<float>(r-1,c+1) > sample);      // UR
            lbp_val[2] = (img.at<float>(r,c+radius) > sample);    // R
            lbp_val[3] = (img.at<float>(r+1,c+1) > sample);       // DR
            lbp_val[4] = (img.at<float>(r+radius,c) > sample);  // D
            lbp_val[5] = (img.at<float>(r+1,c-1) > sample);     // DL
            lbp_val[6] = (img.at<float>(r,c-radius) > sample);   // L
            lbp_val[7] = (img.at<float>(r-1,c-1) > sample);    // UL
            cout<<"Current Position: "<<r<<","<<c<<" val: "<<target<<','<<lbp_val<<endl;
            lbp.at<float>(r-radius,c-radius) = target;
        }
    }
    cout<<"LBP Complete"<<endl;
    */

    /* 圆形LBP特征计算，由于输出数据过于诡异（1.0E-45级别）已弃用
    lbp = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32FC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<float>(i+fy,j+fx) + w2*src.at<float>(i+fy,j+cx) + w3*src.at<float>(i+cy,j+fx) + w4*src.at<float>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                lbp.at<unsigned int>(i-radius,j-radius) += ((t > src.at<float>(i,j)) && (abs(t-src.at<float>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
    */
    getUniformPatternLBPFeature<float>(src, lbp, radius, neighbors);
    // 找最大值
    float max = 0;
    for(int r=0;r<lbp.rows;r++){
        for(int c=0;c<lbp.cols;c++){
            float tmp = lbp.at<float>(r,c);
            if(max<tmp) max = tmp;
        }
    }
    cout<<"Max: "<<max<<endl;
    // int nbin_x = pow(2,neighbors)-1;
    int nbin_x = neighbors*(neighbors-1)+4;
    int nbin_y = 1; 
    int histSize[] = {nbin_x,nbin_y};
    int channles[] = {0};
    // 直方图宽度标定范围
    float axis_z[] = { 0, binsNumber+1 };
    // // 只有一个维度
    // float axis_y[] = { 0, 1 };
    const float* ranges = { axis_z };
    // MatND lbp_hist;
    cout<<"nbin_x: "<<nbin_x<<" ,nbin_y: "<<nbin_y<<endl;
    lbp_hist = Mat::zeros(nbin_x,nbin_y,CV_32FC1);
    calcHist(&lbp, 1, channles, Mat(), lbp_hist, 1, &nbin_x, &ranges);
    MatND norn_hist;
    normalize(lbp_hist, lbp_hist, 0, 1, NORM_MINMAX,-1,Mat());

    cout<<"=============exit lbp_features============="<<endl;
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
    
    // 点测正则化后的face_hist
    cout<<"Normalized_face_hist "<<140<<" : "<<face_hist.at<float>(140,0)<<endl;
    
    // 文件输出以对照结果
    ofstream ofile;
    ofile.open("face_hist.txt",ios::out);
    for(int i=0;i<256;i++){
        ofile<<face_hist.at<float>(i,0)<<endl;
    }
    ofile.close();

    // 提取LBP特征
    MatND lbp;
    MatND lbp_hist;
    int radius = 2;
    int neighbors = 8;
    get_lbp_feature(dst, radius, neighbors, lbp, lbp_hist);

    cout<<"lpb_hist.cols: "<<lbp_hist.cols<<" | lbp_hist.rows: "<<lbp_hist.rows<<endl;
    cout<<"lpb.cols: "<<lbp.cols<<" | lbp.rows: "<<lbp.rows<<endl;

    cout<<"LBP features of img: "<<endl;
    for(int i=0;i<3;i++){
        cout<<lbp.at<float>(i,0)<<" ";
    }
    cout<<endl;
    // 文件输出lbp_hist
    ofile.open("lbp_hist.txt",ios::out);
    for(int i=1;i<lbp_hist.rows;i++){
        ofile<<lbp_hist.at<float>(i,0)<<endl;
    }
    ofile.close();
    // 文件输出lbp
    ofile.open("lbp.txt",ios::out);
    for(int i=0;i<lbp.rows;i++){
        for(int j=0;j<lbp.cols;j++){
            ofile<<lbp.at<float>(i,j)<<" ";
        }
        ofile<<endl;
    }
    ofile.close();
    return 0;
}

//等价模式LBP特征计算
template <typename _tp>
void getUniformPatternLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors)
{
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
int getHopTimes(int n)
{
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
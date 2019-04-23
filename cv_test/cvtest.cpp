// #include <opencv.h>  
#include <opencv2/opencv.hpp>
#include <iostream>

using std::cout;
using std::endl;
using namespace cv;  

int main(int argc, char* argv[])  
{  
    Mat image;  
    image = imread(argv[1]);
    cout<<argv[1]<<endl;
    if(!image.data) {
        cout <<"Read Failed"<<endl;
        return -1;
    }  
    // namedWindow("Display Image", CV_WINDOW_AUTOSIZE);
    Rect area(10,10,100,100);
    Mat region = image(area);
    imshow("Display Image", region);  
    waitKey(0);  
    return 0;  
}

#include "./include/features.h"


int main(int argc, char** argv){
    string img_path = "lena.jpg";
    if(spoof_detect(img_path)) cout<<"Complete"<<endl;
    return 0;
}

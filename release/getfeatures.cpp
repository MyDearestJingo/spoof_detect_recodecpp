#include "./include/features.h"

string img_path = "lena.jpg";

int main(int argc, char** argv){
    if(spoof_detect(img_path)) cout<<"Complete"<<endl;
    return 0;
}
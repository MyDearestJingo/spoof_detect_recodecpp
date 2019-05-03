#include <fstream>
#include <string>
#include <vector>
#include "features.h"

typedef struct data{
    Mat cropped_img;
    int cropped_bbox[4];
}data;

vector<data> v_data;
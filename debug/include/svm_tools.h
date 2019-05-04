#include <fstream>
#include <string>
#include <vector>
#include "features.h"

#define DEBUG

typedef struct data{
    Mat cropped_img;
    int cropped_bbox[4];
}data;

void read_data(ifstream &FILE, string dir, vector<data> &v_data);
void preproc_data(vector<data> &v_data,const int n_samples, float fd[][256*2+59]);

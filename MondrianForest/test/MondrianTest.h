#pragma once
#include <string>
#include <iostream>
#include <fstream>
using std::string;
using std::cout;
using std::endl;
using std::ifstream;
#include "../ml/MondrianForest.h"
using cv::Algorithm;
using cv::ml::OnlineForest;
using cv::ml::MondrianForest;
using cv::ml::TrainData;
using cv::ml::ROW_SAMPLE;
using cv::Ptr;
using cv::Mat;
class MondrianTest
{
public:
	void trainAndTest(const string &train_file, const string &test_file);
private:
	int loadData(const string &file, Ptr< TrainData > *data);
};


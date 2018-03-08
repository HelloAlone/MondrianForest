#include "MondrianTest.h"



void MondrianTest::trainAndTest(const string & train_file, const string & test_file)
{
	Ptr<TrainData> train_data, test_data;
	int n_class_train = loadData(train_file, &train_data);
	int n_class_test = loadData(test_file, &test_data);
	CV_Assert(n_class_train == n_class_test);

	Ptr< MondrianForest > forest = MondrianForest::create();
	forest->setTreeNumber(20);
	MondrianForest::MondrianParams params;
	params.number_of_classes = n_class_train;
	forest->setMParams(params);
	double t = (double)cv::getTickCount();
	forest->train(train_data);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "train, t = " << t * 1000 << " ms" << endl;
	cout << "trained!" << endl;
	t = (double)cv::getTickCount();
	forest->extend(test_data);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "extend, t = " << t * 1000 << " ms" << endl;
	cout << "extended!" << endl;
	//forest->save("mf.xml");

	//Ptr< OnlineForest > forest = Algorithm::load<MondrianForest>("mf.xml");
	//cout << "forest loaded!" << endl;

	Mat results;
	vector<int> ground_truth = test_data->getTrainResponses();
	int right = 0;
	forest->predict(test_data->getTrainSamples(), results);
	for (int _row = 0; _row < results.rows; _row++)
	{
		float max_predict_value = FLT_MIN;
		int predict_class = 0;
		for (int _col = 0; _col < results.cols; _col++)
		{
			if (results.at<float>(_row, _col) > max_predict_value)
			{
				max_predict_value = results.at<float>(_row, _col);
				predict_class = _col;
			}
		}
		cout << predict_class << " " << ground_truth[_row] << " " << (predict_class == ground_truth[_row]) << endl;
		if (predict_class == ground_truth[_row])
		{
			right++;
		}
	}
	cout << (double)right / test_data->getNTrainSamples() << endl;
}

int MondrianTest::loadData(const string & file, Ptr<TrainData> *data)
{
	ifstream ifs;
	ifs.open(file);
	if (ifs.is_open())
	{
		// read n, n_dim, n_class
		int n, n_dim, n_class;
		ifs >> n_dim >> n >> n_class;
		// prepare
		Mat samples(n, n_dim, CV_32F);
		//vector<int> responses(n);
		Mat responses(n, 1, CV_32S);
		// read samples
		for (int i = 0; i < n; i++)
		{
			for (int d = 0; d < n_dim; d++)
			{
				ifs >> samples.at<float>(i, d);
			}
		}
		// read response
		for (int d = 0; d < n; d++)
		{
			//ifs >> responses[d];
			ifs >> responses.at<int>(d, 0);
		}
		// close
		ifs.close();
		cout << "file loaded" << endl;
		// create data
		*data = TrainData::create(samples, ROW_SAMPLE, responses);
		cout << "data created!" << endl;
		return n_class;
	}
	else
	{
		return 0;
	}
}


#pragma once
#include <opencv2/ml/ml.hpp>
#include <vector>
using std::vector;
#include <iostream>
using std::cout;
using std::endl;
#ifdef HAVE_TBB
#include <tbb/tbb.h>
#endif // HAVE_TBB

namespace cv
{
	namespace ml
	{
		/****************************************************************************************\
		*                                      Forest                                           *
		\****************************************************************************************/
		class Forest :
			public StatModel
		{
		protected:
			virtual float predictTree(InputArray samples, int tree, OutputArray results = noArray(), int flags = 0) const = 0;
			virtual bool trainTree(const Ptr< TrainData > &trainData, int tree, int flags = 0) = 0;
		public:
			virtual float predict(InputArray samples, OutputArray results = noArray(), int flags = 0) const;
			virtual bool train(const Ptr< TrainData > &trainData, int flags = 0);
		public:
			inline virtual int getTreeNumber() { return tree_number; }
			inline virtual void setTreeNumber(int val) 
			{
				if (val < 0)
				{
					CV_Error(CV_StsOutOfRange, "tree_number should be >= 0");
				}
				tree_number = val;
			}
		public:
			Forest() { tree_number = 20; }
		protected:
			int tree_number;
			vector<int> roots;
		};
		/****************************************************************************************\
		*                                      OnlineForest                                     *
		\****************************************************************************************/
		class OnlineForest :
			public Forest
		{
		protected:
			virtual bool extendTree(const Ptr< TrainData > &trainData, int tree, int flags = 0) = 0;
		public:
			virtual bool extend(const Ptr< TrainData > &trainData, int flags = 0);
		};
	}
}

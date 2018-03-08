#include "Forest.h"



namespace cv
{
	namespace ml
	{
		float Forest::predict(InputArray samples, OutputArray results, int flags) const
		{
			vector<Mat> results_all(tree_number);
#ifdef HAVE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, tree_number),
				[&](tbb::blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i != r.end(); i++)
				{
					predictTree(samples, i, results_all[i], flags);
				}
			});
#else
			for (int tree = 0; tree < tree_number; tree++)
			{
				predictTree(samples, tree, results_all[tree], flags);
			}
#endif
			results_all[0].copyTo(results);
			Mat resultMat = results.getMat();
			for (int tree = 1; tree < tree_number; tree++)
			{
				resultMat += results_all[tree];
			}
			resultMat /= tree_number;

			return 0.0f;
		}

		bool Forest::train(const Ptr<TrainData>& trainData, int flags)
		{
			roots.resize(tree_number);
#ifdef HAVE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, tree_number),
				[&](tbb::blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i != r.end(); i++)
				{
					trainTree(trainData, i, flags);
				}
			});
#else
			for (int tree = 0; tree < tree_number; tree++)
			{
				if (!trainTree(trainData, tree, flags))
					return false;
			}
#endif
			return true;
		}

		bool OnlineForest::extend(const Ptr<TrainData>& trainData, int flags)
		{
			CV_Assert(isTrained());
#ifdef HAVE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, tree_number),
				[&](tbb::blocked_range<size_t>& r)
			{
				for (int i = r.begin(); i != r.end(); i++)
				{
					extendTree(trainData, i, flags);
				}
			});
#else
			for (int tree = 0; tree < tree_number; tree++)
			{
				if (!extendTree(trainData, tree, flags))
					return false;
			}
#endif
			return true;
		}
	}
}

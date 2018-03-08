#pragma once
#include "Forest.h"
#include <queue>
#include <stack>
#include <algorithm>
#include <random>
using std::queue;
using std::stack;
using std::pair;
using std::min;
using std::max;
using std::random_device;
using std::mt19937;
using std::exponential_distribution;
using std::uniform_real_distribution;
using std::numeric_limits;
namespace cv
{
	namespace ml
	{
		/****************************************************************************************\
		*                                      MondrianForest                                   *
		\****************************************************************************************/

		class MondrianForest :
			public OnlineForest
		{
		public:
			struct MondrianParams
			{
				MondrianParams(float _lifetime = numeric_limits<float>::infinity(), float _discount_parameter = 10, int _number_of_classes = 2)
				{
					lifetime = _lifetime;
					discount_parameter = _discount_parameter;
					number_of_classes = _number_of_classes;
					dimension = 0;
				}
				float lifetime;
				float discount_parameter;
				int number_of_classes;
				int dimension;
			};
		protected:
			struct MondrianBlock
			{
				MondrianBlock(int _this_, int _parent)
				{ 
					this_ = _this_;
					parent = _parent;
					left_child = right_child = -1;
					is_leaf = false;
				}
				float time_of_split;
				int split_dimension;
				float split_location;
				bool is_leaf;
				int left_child;
				int right_child;
				int parent;
				int this_;
				vector<float> lower_bounds;
				vector<float> upper_bounds;
				vector<int> c;
				vector<int> tab;
				vector<float> G;
			};
		public:
			virtual void clear();
			virtual String getDefaultName() const;
			virtual int getVarCount() const;
			virtual bool isClassifier() const;
			virtual bool isTrained() const;
			virtual void write(FileStorage &fs) const;
			virtual void read(const FileNode& fn);
			virtual bool train(const Ptr< TrainData > &trainData, int flags = 0);
			virtual bool extend(const Ptr< TrainData > &trainData, int flags = 0);
			virtual float predict(InputArray samples, OutputArray results = noArray(), int flags = 0) const;
		protected:
			virtual float predictTree(InputArray samples,int tree, OutputArray results = noArray(), int flags = 0) const;
			virtual void writeParams(FileStorage& fs) const;
			virtual void writeTree(FileStorage& fs, int tree) const;
			virtual void readParams(const FileNode& fn);
			virtual void readTree(const FileNode& fn, int tree);
			virtual bool trainTree(const Ptr< TrainData > &trainData, int tree, int flags = 0);
		protected:
			virtual bool extendTree(const Ptr< TrainData > &trainData, int tree, int flags = 0);
		protected:
			using node_status = pair < int, vector <int> >;
			void sampleMondrianBlock(int tree, int start, const vector<int> &data);
			void extendMondrianBlock(int tree, int start, const vector<int> &data);
			int sampleSplitDimension(const std::vector<double> &range, mt19937 &gen);
			//void initializePosteriorCounts(int tree, int leaf, const vector<int> &data);
			//void updatePosteriorCounts(int tree, int leaf, const vector<int> &data);
			void computeLeafPosteriorCounts(int tree, int leaf, const vector<int> &data);
			void computePosteriorCounts(int tree);
			void computePosteriorPredictiveDistribution(int tree);
			const vector<int> & calcCAndReturnTab(int tree, int j);
		public:
			MondrianForest();
			virtual ~MondrianForest();
		public:
			inline MondrianParams getMParams() { return params; }
			inline void setMParams(const MondrianParams& _params)
			{ 
				params = _params;
				base_distribution = 1.0f / params.number_of_classes;
			}
		protected:
			int var_count;
			MondrianParams params;
			float base_distribution;
			vector< vector< MondrianBlock > > blocks;
			Mat X;
			vector<int> Y;
			random_device rd;
		public:
			static Ptr< MondrianForest > create();
		};
	}
}

#include "MondrianForest.h"



namespace cv
{
	namespace ml
	{
		void MondrianForest::clear()
		{
			var_count = 0;
			roots.clear();
			for (auto t : blocks)
			{
				t.clear();
			}
			blocks.clear();
			X.release();
			Y.clear();
		}

		String MondrianForest::getDefaultName() const
		{
			return "opencv_ml_mondrianforest";
		}

		int MondrianForest::getVarCount() const
		{
			return var_count;
		}

		bool MondrianForest::isClassifier() const
		{
			// 没有实现除分类外的其他功能
			return true;
		}

		bool MondrianForest::isTrained() const
		{
			if (roots.empty())
			{
				return false;
			}
			return true;
		}

		void MondrianForest::write(FileStorage & fs) const
		{
			writeParams(fs);
			fs << "trees" << "[";
			for (int tree = 0; tree < tree_number; tree++)
			{
				writeTree(fs, tree);
			}
			fs << "]";
		}

		void MondrianForest::read(const FileNode & fn)
		{
			clear();
			readParams(fn);
			FileNode &trees = fn["trees"];
			CV_Assert(trees.isSeq() && trees.size() == tree_number);
			roots.resize(tree_number);
			blocks.resize(tree_number);
			for (int tree = 0; tree < tree_number; tree++)
			{
				readTree(trees, tree);
			}
		}

		bool MondrianForest::train(const Ptr<TrainData>& trainData, int flags)
		{
			clear();
			// 只支持0~k-1的class label
			Mat responses = trainData->getTrainResponses();
			if (responses.type() != CV_32S)
			{
				CV_Error(CV_StsBadArg, "responses type must be CV_32S");
			}
			// 每行是一个样本
			X = trainData->getTrainSamples(ROW_SAMPLE);
			responses.copyTo(Y);
			// 初始化
			params.dimension = X.cols;
			var_count = X.rows;
			blocks.resize(tree_number);
			for (auto b : blocks)
			{
				b.reserve(var_count);
			}

			bool res = Forest::train(trainData, flags);
			if (res)
			{
				X.release();
				Y.clear();
			}
			else
			{
				clear();
			}
			return res;
		}

		bool MondrianForest::extend(const Ptr<TrainData>& trainData, int flags)
		{
			// 只支持0~k-1的class label
			Mat responses = trainData->getTrainResponses();
			if (responses.type() != CV_32S)
			{
				CV_Error(CV_StsBadArg, "responses type must be CV_32S");
			}
			// 每行是一个样本
			X = trainData->getTrainSamples(ROW_SAMPLE);
			responses.copyTo(Y);

			CV_Assert(params.dimension == X.cols);
			var_count += X.rows;

			bool res = OnlineForest::extend(trainData, flags);
			if (res)
			{
				X.release();
				Y.clear();
			}
			else
			{
				clear();
			}
			return res;
		}

		float MondrianForest::predict(InputArray samples, OutputArray results, int flags) const
		{
			CV_Assert(samples.type() == CV_32F);
			CV_Assert(results.needed());
			CV_Assert(samples.cols() == params.dimension);
			return Forest::predict(samples, results, flags);
		}

		float MondrianForest::predictTree(InputArray samples, int tree, OutputArray results, int flags) const
		{
			Mat sampleMat = samples.getMat();
			results.create(sampleMat.rows, params.number_of_classes, CV_32F);
			Mat resultMat = results.getMat();
			resultMat = Scalar_<float>(0.0f);
			vector<float> p_not_separated_yet(sampleMat.rows, 1.0f);

			vector<int> all_data_(sampleMat.rows);
			for (auto i = 0; i < all_data_.size(); i++)
			{
				all_data_[i] = i;
			}
			queue<node_status> queue_;
			queue_.emplace(node_status(roots[tree], all_data_));
			while (!queue_.empty())
			{
				node_status &node = queue_.front();
				int j = node.first;
				const MondrianBlock &block = blocks[tree][j];
				vector<int> &data_ = node.second;
				if (data_.size() < 1)
				{
					continue;
				}
				vector<int> data_left, data_right;
				for (int _row : data_)
				{
					float delta = block.this_ == roots[tree] ? block.time_of_split :
						block.time_of_split - blocks[tree][block.parent].time_of_split;
					float eta = 0.0f;
					for (int _d = 0; _d < params.dimension; _d++)
					{
						eta += max(sampleMat.at<float>(_row, _d) - block.upper_bounds[_d], 0.0f) +
							max(block.lower_bounds[_d] - sampleMat.at<float>(_row, _d), 0.0f);
					}
					float p = abs(eta) <= FLT_EPSILON ? 0.0f : 1 - exp(-delta * eta);
					if (p > 0.0f)
					{
						float discount = (eta / (eta + params.discount_parameter)) *
							(-expm1(-(eta + params.discount_parameter) * delta) /
								(-expm1(-eta * delta)));
						int num_customers = 0;
						for (int _c : block.c)
						{
							num_customers += _c;
						}
						float discount_per_customers = discount / num_customers;
						vector<int> c_(params.number_of_classes);
						vector<int> tab_(params.number_of_classes);
						vector<float> G_(params.number_of_classes);
						float c_sum = 0.0f;
						float tab_sum = 0.0f;
						for (int k = 0; k < params.number_of_classes; k++)
						{
							c_[k] = tab_[k] = min(block.c[k], 1);
							c_sum += (float)c_[k];
							tab_sum += (float)tab_[k];
						}
						if (block.this_ == roots[tree])
						{
							for (int k = 0; k < params.number_of_classes; k++)
							{
								G_[k] = (c_[k] - discount_per_customers * tab_[k] +
									discount_per_customers * tab_sum * base_distribution) / c_sum;
								resultMat.at<float>(_row, k) += p_not_separated_yet[_row] * p * G_[k];
							}
						}
						else
						{
							const MondrianBlock &parent = blocks[tree][block.parent];
							for (int k = 0; k < params.number_of_classes; k++)
							{
								G_[k] = (c_[k] - discount_per_customers * tab_[k] +
									discount_per_customers * tab_sum * parent.G[k]) / c_sum;
								resultMat.at<float>(_row, k) += p_not_separated_yet[_row] * p * G_[k];
							}
						}
					}
					if (block.is_leaf)
					{
						for (int k = 0; k < params.number_of_classes; k++)
						{
							resultMat.at<float>(_row, k) += p_not_separated_yet[_row] * (1 - p) * block.G[k];
						}
					}
					else
					{
						p_not_separated_yet[_row] = p_not_separated_yet[_row] * (1 - p);
						if (sampleMat.at<float>(_row, block.split_dimension) > block.split_location)
						{
							data_right.push_back(_row);
						}
						else
						{
							data_left.push_back(_row);
						}
					}
				}
				queue_.pop();
				if (block.left_child >= 0 && data_left.size() > 0)
				{
					queue_.emplace(node_status(block.left_child, data_left));
				}
				if (block.right_child >= 0 && data_right.size() > 0)
				{
					queue_.emplace(node_status(block.right_child, data_right));
				}
			}

			return 0.0f;
		}

		void MondrianForest::writeParams(FileStorage & fs) const
		{
			fs << "tree_number" << tree_number;
			fs << "var_count" << var_count;
			// MondrianParams
			fs << "mondrian_params" << "{";
			fs << "life_time" << params.lifetime;
			fs << "discount_parameter" << params.discount_parameter;
			fs << "number_of_classes" << params.number_of_classes;
			fs << "dimension" << params.dimension;
			fs << "}";
		}

		void MondrianForest::writeTree(FileStorage & fs, int tree) const
		{
			fs << "{";
			// root
			fs << "root" << roots[tree];
			// blocks
			fs << "blocks" << "[";
			for (size_t i = 0; i < blocks[tree].size(); i++)
			{
				const MondrianBlock &block = blocks[tree][i];
				// one block
				fs << "{";
				fs << "time_of_split" << block.time_of_split;
				fs << "split_dimension" << block.split_dimension;
				fs << "split_location" << block.split_location;
				fs << "is_leaf" << block.is_leaf;
				fs << "left_child" << block.left_child;
				fs << "right_child" << block.right_child;
				fs << "parent" << block.parent;
				fs << "this_" << block.this_;

				fs << "lower_bounds" << "[";
				for (size_t j = 0; j < block.lower_bounds.size(); j++)
				{
					fs << block.lower_bounds[j];
				}
				fs << "]";

				fs << "upper_bounds" << "[";
				for (size_t j = 0; j < block.upper_bounds.size(); j++)
				{
					fs << block.upper_bounds[j];
				}
				fs << "]";

				fs << "c" << "[";
				for (size_t j = 0; j < block.c.size(); j++)
				{
					fs << block.c[j];
				}
				fs << "]";

				fs << "tab" << "[";
				for (size_t j = 0; j < block.tab.size(); j++)
				{
					fs << block.tab[j];
				}
				fs << "]";

				fs << "G" << "[";
				for (size_t j = 0; j < block.G.size(); j++)
				{
					fs << block.G[j];
				}
				fs << "]";

				fs << "}";
			}
			fs << "]";
			fs << "}";
		}

		void MondrianForest::readParams(const FileNode & fn)
		{
			tree_number = (int)fn["tree_number"];
			var_count = (int)fn["var_count"];
			FileNode &mondrian_params = fn["mondrian_params"];
			CV_Assert(mondrian_params.isMap());
			params.lifetime = (float)mondrian_params["life_time"];
			params.discount_parameter = (float)mondrian_params["discount_parameter"];
			params.number_of_classes = (int)mondrian_params["number_of_classes"];
			params.dimension = (int)mondrian_params["dimension"];
			base_distribution = 1.0f / params.number_of_classes;
		}

		void MondrianForest::readTree(const FileNode &fn, int tree)
		{
			FileNode &one_tree = fn[tree];
			CV_Assert(one_tree.isMap());
			roots[tree] = (int)one_tree["root"];
			FileNode &one_tree_blocks = one_tree["blocks"];
			CV_Assert(one_tree_blocks.isSeq() && one_tree_blocks.size() > 0);
			blocks[tree].resize(one_tree_blocks.size(), MondrianBlock(-1, -1));
			for (size_t i = 0; i < one_tree_blocks.size(); i++)
			{
				FileNode &one_block = one_tree_blocks[i];
				CV_Assert(one_block.isMap());
				MondrianBlock &block = blocks[tree][i];
				block.time_of_split = (float)one_block["time_of_split"];
				block.split_dimension = (int)one_block["split_dimension"];
				block.split_location = (float)one_block["split_location"];
				block.is_leaf = (bool)(int)one_block["is_leaf"];
				block.left_child = (int)one_block["left_child"];
				block.right_child = (int)one_block["right_child"];
				block.parent = (int)one_block["parent"];
				block.this_ = (int)one_block["this_"];

				FileNode &one_block_lower_bounds = one_block["lower_bounds"];
				CV_Assert(one_block_lower_bounds.isSeq() && one_block_lower_bounds.size() == params.dimension);
				block.lower_bounds.resize(params.dimension);
				for (size_t j = 0; j < one_block_lower_bounds.size(); j++)
				{
					block.lower_bounds[j] = (float)one_block_lower_bounds[j];
				}

				FileNode &one_block_upper_bounds = one_block["upper_bounds"];
				CV_Assert(one_block_upper_bounds.isSeq() && one_block_upper_bounds.size() == params.dimension);
				block.upper_bounds.resize(params.dimension);
				for (size_t j = 0; j < one_block_upper_bounds.size(); j++)
				{
					block.upper_bounds[j] = (float)one_block_upper_bounds[j];
				}

				FileNode &one_block_c = one_block["c"];
				CV_Assert(one_block_c.isSeq() && one_block_c.size() == params.number_of_classes);
				block.c.resize(params.number_of_classes);
				for (size_t j = 0; j < one_block_c.size(); j++)
				{
					block.c[j] = one_block_c[j];
				}

				FileNode &one_block_tab = one_block["tab"];
				CV_Assert(one_block_tab.isSeq() && one_block_tab.size() == params.number_of_classes);
				block.tab.resize(params.number_of_classes);
				for (size_t j = 0; j < one_block_tab.size(); j++)
				{
					block.tab[j] = one_block_tab[j];
				}

				FileNode &one_block_G = one_block["G"];
				CV_Assert(one_block_G.isSeq() && one_block_G.size() == params.number_of_classes);
				block.G.resize(params.number_of_classes);
				for (size_t j = 0; j < one_block_G.size(); j++)
				{
					block.G[j] = (float)one_block_G[j];
				}
			}
		}

		bool MondrianForest::trainTree(const Ptr<TrainData>& trainData, int tree, int flags)
		{
			roots[tree] = 0;
			vector<int> all_data_(X.rows);
			for (auto i = 0; i < all_data_.size(); i++)
			{
				all_data_[i] = i;
			}
			// root的父亲是-1，包含全部数据
			sampleMondrianBlock(tree, -1, all_data_);

			computePosteriorCounts(tree);
			computePosteriorPredictiveDistribution(tree);

			return true;
		}

		bool MondrianForest::extendTree(const Ptr<TrainData>& trainData, int tree, int flags)
		{
			vector<int> all_data_(X.rows);
			for (auto i = 0; i < all_data_.size(); i++)
			{
				all_data_[i] = i;
			}
			// 从root开始
			extendMondrianBlock(tree, roots[tree], all_data_);

			computePosteriorCounts(tree);
			computePosteriorPredictiveDistribution(tree);

			return true;
		}

		void MondrianForest::sampleMondrianBlock(int tree, int start, const vector<int>& data)
		{
			// 相比原版的SampleMondrianTree，改用非递归的方式生成树
			// 初始化随机数发生器
			mt19937 gen(rd() + tree);
			// 从start开始扩展子树
			queue< node_status > queue_;
			queue_.emplace(node_status(start, data));
			while (!queue_.empty())
			{
				node_status &node = queue_.front();
				int parent_ = node.first;
				vector<int> &data_ = node.second;
				CV_Assert(data_.size() > 0);
				blocks[tree].emplace_back(MondrianBlock(blocks[tree].size(), parent_));
				MondrianBlock &block = blocks[tree].back();
				CV_Assert(block.this_ == blocks[tree].size() - 1);
				if (block.parent >= 0)
				{
					MondrianBlock &block_parent = blocks[tree][block.parent];
					if (block_parent.left_child < 0)
					{
						block_parent.left_child = block.this_;
					}
					else
					{
						block_parent.right_child = block.this_;
					}
				}
				block.lower_bounds = vector<float>(params.dimension, numeric_limits<float>::infinity());
				block.upper_bounds = vector<float>(params.dimension, -numeric_limits<float>::infinity());
				for (int _row : data_)
				{
					for (int _col = 0; _col < X.cols; _col++)
					{
						block.lower_bounds[_col]
							= min(block.lower_bounds[_col],
								X.at<float>(_row, _col));
						block.upper_bounds[_col]
							= max(block.upper_bounds[_col],
								X.at<float>(_row, _col));
					}
				}
				vector<double> range(params.dimension, 0);
				double rate = 0;
				for (int _d = 0; _d < params.dimension; _d++)
				{
					range[_d] = block.upper_bounds[_d] - block.lower_bounds[_d];
					rate += range[_d];
				}
				exponential_distribution<double> d(rate);
				// E，rate = 0时返回inf
				float E = (float)d(gen);
				block.time_of_split = block.this_ == roots[tree] ? E :
					blocks[tree][block.parent].time_of_split + E;
				if (block.time_of_split < params.lifetime)
				{
					// split dimension
					block.split_dimension = sampleSplitDimension(range, gen);
					// split location
					uniform_real_distribution<float> distribution(
						block.lower_bounds[block.split_dimension] + FLT_EPSILON,
						block.upper_bounds[block.split_dimension] - FLT_EPSILON);
					block.split_location = distribution(gen);
					vector<int> data_left, data_right;
					for (int _row : data_)
					{
						if (X.at<float>(_row, block.split_dimension) > block.split_location)
						{
							data_right.push_back(_row);
						}
						else
						{
							data_left.push_back(_row);
						}
					}
					CV_Assert(data_left.size() > 0);
					CV_Assert(data_right.size() > 0);
					queue_.pop();
					queue_.emplace(node_status(block.this_, data_left));
					queue_.emplace(node_status(block.this_, data_right));
				}
				else
				{
					block.time_of_split = params.lifetime;
					block.is_leaf = true;
					//initializePosteriorCounts(tree, block.this_, data_);
					computeLeafPosteriorCounts(tree, block.this_, data_);
					queue_.pop();
				}
			}
		}

		void MondrianForest::extendMondrianBlock(int tree, int start, const vector<int>& data)
		{
			// 初始化随机数发生器
			mt19937 gen(rd() + tree);
			// 从start开始
			queue<node_status> queue_;
			queue_.emplace(node_status(start, data));

			while (!queue_.empty())
			{
				node_status &node = queue_.front();
				MondrianBlock &block = blocks[tree][node.first];
				vector<int> &data_ = node.second;
				CV_Assert(data_.size() > 0);
				vector<float> el(params.dimension, 0);
				vector<float> eu(params.dimension, 0);
				for (int _row : data_)
				{
					for (int _col = 0; _col < X.cols; _col++)
					{
						el[_col] = max(el[_col], max(block.lower_bounds[_col]
							- X.at<float>(_row, _col), 0.0f));
						eu[_col] = max(eu[_col], max(X.at<float>(_row, _col)
							- block.upper_bounds[_col], 0.0f));
					}
				}
				vector<double> range(params.dimension, 0);
				double rate = 0;
				for (int _d = 0; _d < params.dimension; _d++)
				{
					range[_d] = el[_d] + eu[_d];
					rate += range[_d];
				}
				exponential_distribution<double> d(rate);
				// E
				float E = (float)d(gen);
				float time_of_split_parent = block.this_ == roots[tree] ? 0.0f : blocks[tree][block.parent].time_of_split;
				if (time_of_split_parent + E < block.time_of_split)
				{
					// split dimension
					int split_dimension = sampleSplitDimension(range, gen);
					// split location
					float split_location;
					bool draw_from_lower = (float)rand() / RAND_MAX <=
						el[split_dimension] / (el[split_dimension] + eu[split_dimension]);
					if (draw_from_lower)
					{
						uniform_real_distribution<float> distribution(
							block.lower_bounds[split_dimension] - el[split_dimension] + FLT_EPSILON,
							block.upper_bounds[split_dimension] - FLT_EPSILON);
						split_location = distribution(gen);
					}
					else
					{
						uniform_real_distribution<float> distribution(
							block.upper_bounds[split_dimension] + FLT_EPSILON,
							block.upper_bounds[split_dimension] + eu[split_dimension] - FLT_EPSILON);
						split_location = distribution(gen);
					}
					// insert new node 
					blocks[tree].emplace_back(MondrianBlock(blocks[tree].size(), block.parent));
					MondrianBlock &new_block = blocks[tree].back();
					new_block.split_dimension = split_dimension;
					new_block.split_location = split_location;
					MondrianBlock &block = blocks[tree][node.first];
					block.parent = new_block.this_;
					if (new_block.parent < 0)	// new root
					{
						roots[tree] = new_block.this_;
						new_block.time_of_split = E;
					}
					else
					{
						MondrianBlock &parent = blocks[tree][new_block.parent];
						if (parent.left_child == block.this_)
						{
							parent.left_child = new_block.this_;
						}
						else
						{
							parent.right_child = new_block.this_;
						}
						new_block.time_of_split = parent.time_of_split + E;
					}
					new_block.lower_bounds.resize(params.dimension);
					new_block.upper_bounds.resize(params.dimension);
					for (int _d = 0; _d < params.dimension; _d++)
					{
						new_block.lower_bounds[_d] = block.lower_bounds[_d] - el[_d];
						new_block.upper_bounds[_d] = block.upper_bounds[_d] + eu[_d];
					}
					if (draw_from_lower)
					{
						new_block.right_child = block.this_;
					}
					else
					{
						new_block.left_child = block.this_;
					}
					vector<int> data_left, data_right;
					for (int _row : data_)
					{
						if (X.at<float>(_row, split_dimension) > split_location)
						{
							data_right.emplace_back(_row);
						}
						else
						{
							data_left.emplace_back(_row);
						}
					}
					if (new_block.left_child < 0)
					{
						CV_Assert(data_left.size() > 0);
						sampleMondrianBlock(tree, new_block.this_, data_left);
						int block_this_ = node.first;
						queue_.pop();
						if (data_right.size() > 0)
						{
							queue_.emplace(node_status(block_this_, data_right));
						}
					}
					else
					{
						CV_Assert(data_right.size() > 0);
						sampleMondrianBlock(tree, new_block.this_, data_right);
						int block_this_ = node.first;
						queue_.pop();
						if (data_left.size() > 0)
						{
							queue_.emplace(node_status(block_this_, data_left));
						}
					}
				}
				else
				{
					for (int _d = 0; _d < params.dimension; _d++)
					{
						block.lower_bounds[_d] = block.lower_bounds[_d] - el[_d];
						block.upper_bounds[_d] = block.upper_bounds[_d] + eu[_d];
					}
					if (!block.is_leaf)
					{
						vector<int> data_left, data_right;
						for (int _row : data_)
						{
							if (X.at<float>(_row, block.split_dimension) > block.split_location)
							{
								data_right.emplace_back(_row);
							}
							else
							{
								data_left.emplace_back(_row);
							}
						}
						queue_.pop();
						if (data_left.size() > 0)
						{
							queue_.emplace(node_status(block.left_child, data_left));
						}
						if (data_right.size() > 0)
						{
							queue_.emplace(node_status(block.right_child, data_right));
						}
					}
					else
					{
						computeLeafPosteriorCounts(tree, block.this_, data_);
						queue_.pop();
					}
				}
			}
		}

		int MondrianForest::sampleSplitDimension(const std::vector<double>& range, mt19937 &gen)
		{
			CV_Assert(range.size() > 0);
			// cumulative sum
			vector<double> cumsum;
			cumsum.resize(range.size());
			cumsum[0] = range[0];
			for (int d = 1; d < range.size(); d++)
			{
				cumsum[d] = range[d] + cumsum[d - 1];
			}
			//double s = max(((double)rand() / RAND_MAX) * cumsum.back(), DBL_EPSILON);
			uniform_real_distribution<double> distribution(0.0 + DBL_EPSILON, cumsum.back() - DBL_EPSILON);
			double s = distribution(gen);
			int k = 0;
			for (vector<double>::iterator it = cumsum.begin(); it != cumsum.end(); it++)
			{
				if (s > (*it))
				{
					k++;
				}
				else
				{
					break;
				}
			}
			return k;
		}

		//void MondrianForest::initializePosteriorCounts(int tree, int leaf, const vector<int>& data)
		//{
		//	MondrianBlock &j = blocks[tree][leaf];
		//	j.c.resize(params.number_of_classes, 0);
		//	j.tab.resize(params.number_of_classes, 0);
		//	for (int _row : data)
		//	{
		//		int k = Y[_row];
		//		CV_Assert(k >= 0 && k < params.number_of_classes);
		//		j.c[k]++;
		//	}
		//	while (true)
		//	{
		//		if (!j.is_leaf)
		//		{
		//			if (j.left_child < 0 || j.right_child < 0)
		//			{
		//				return;
		//			}
		//			j.c.resize(params.number_of_classes, 0);
		//			j.tab.resize(params.number_of_classes, 0);
		//			MondrianBlock &left_ = blocks[tree][j.left_child];
		//			MondrianBlock &right_ = blocks[tree][j.right_child];
		//			if (left_.tab.size() < 1 || right_.tab.size() < 1)
		//			{
		//				return;
		//			}
		//			for (int k = 0; k < params.number_of_classes; k++)
		//			{
		//				j.c[k] = left_.tab[k] + right_.tab[k];
		//			}
		//		}
		//		for (int k = 0; k < params.number_of_classes; k++)
		//		{
		//			j.tab[k] = min(j.c[k], 1);
		//		}
		//		if (j.this_ == roots[tree])
		//		{
		//			return;
		//		}
		//		else
		//		{
		//			j = blocks[tree][j.parent];
		//		}
		//	}
		//}

		//void MondrianForest::updatePosteriorCounts(int tree, int leaf, const vector<int>& data)
		//{
		//	MondrianBlock &j = blocks[tree][leaf];
		//	for (int _row : data)
		//	{
		//		int y = Y[_row];
		//		CV_Assert(y >= 0 && y < params.number_of_classes);
		//		j.c[y]++;
		//	}
		//	while (true)
		//	{
		//		for (int y = 0; y < params.number_of_classes; y++)
		//		{
		//			if (j.c[y] == 0 || j.tab[y] == 1)
		//			{
		//				continue;
		//			}
		//			else
		//			{
		//				MondrianBlock &left_ = blocks[tree][j.left_child];
		//				MondrianBlock &right_ = blocks[tree][j.right_child];
		//				if (!j.is_leaf)
		//				{
		//					j.c[y] = left_.tab[y] + right_.tab[y];
		//				}
		//				j.tab[y] = min(j.c[y], 1);
		//				if (j.this_ == roots[tree])
		//				{
		//					return;
		//				}
		//				else
		//				{
		//					j = blocks[tree][j.parent];
		//				}
		//			}
		//		}
		//	}
		//}

		void MondrianForest::computeLeafPosteriorCounts(int tree, int leaf, const vector<int>& data)
		{
			MondrianBlock &j = blocks[tree][leaf];
			CV_Assert(j.is_leaf);
			if (j.c.size() == 0)
			{
				j.c.resize(params.number_of_classes, 0);
				j.tab.resize(params.number_of_classes, 0);
			}
			for (int _row : data)
			{
				int k = Y[_row];
				CV_Assert(k >= 0 && k < params.number_of_classes);
				j.c[k]++;
			}
			for (int k = 0; k < params.number_of_classes; k++)
			{
				j.tab[k] = min(j.c[k], 1);
			}
		}

		void MondrianForest::computePosteriorCounts(int tree)
		{
			calcCAndReturnTab(tree, roots[tree]);
		}

		void MondrianForest::computePosteriorPredictiveDistribution(int tree)
		{
			queue<int> J;
			J.push(roots[tree]);
			while (!J.empty())
			{
				int j = J.front();
				J.pop();
				MondrianBlock &block = blocks[tree][j];
				block.G.resize(params.number_of_classes);
				CV_Assert(block.c.size() > 0 && block.tab.size() > 0);
				float c_sum = 0.0f;
				float tab_sum = 0.0f;
				for (int k = 0; k < params.number_of_classes; k++)
				{
					c_sum += (float)block.c[k];
					tab_sum += (float)block.tab[k];
				}
				if (block.this_ == roots[tree])
				{
					float d = exp(-params.discount_parameter*(block.time_of_split));
					for (int k = 0; k < params.number_of_classes; k++)
					{
						block.G[k] = (block.c[k] - d * block.tab[k] +
							d * tab_sum * base_distribution) / c_sum;
					}
				}
				else
				{
					MondrianBlock &parent = blocks[tree][block.parent];
					float d = exp(-params.discount_parameter*(block.time_of_split -
						parent.time_of_split));
					for (int k = 0; k < params.number_of_classes; k++)
					{
						block.G[k] = (block.c[k] - d * block.tab[k] +
							d * tab_sum * parent.G[k]) / c_sum;
					}
				}
				if (!block.is_leaf)
				{
					J.push(block.left_child);
					J.push(block.right_child);
				}
			}
		}

		const vector<int> & MondrianForest::calcCAndReturnTab(int tree, int j)
		{
			MondrianBlock &block = blocks[tree][j];
			if (!block.is_leaf)
			{
				if (block.c.size() == 0)
				{
					block.c.resize(params.number_of_classes);
					block.tab.resize(params.number_of_classes);
				}
				const vector<int> &tab_left = calcCAndReturnTab(tree, block.left_child);
				const vector<int> &tab_right = calcCAndReturnTab(tree, block.right_child);
				for (int i = 0; i < params.number_of_classes; i++)
				{
					block.c[i] = tab_left[i] + tab_right[i];
					block.tab[i] = min(block.c[i], 1);
				}
			}
			return block.tab;
		}

		MondrianForest::MondrianForest()
		{
			var_count = 0;
			base_distribution = 1.0f / params.number_of_classes;
		}


		MondrianForest::~MondrianForest()
		{
		}

		Ptr< MondrianForest > MondrianForest::create()
		{
			return makePtr< MondrianForest >();
		}
	}
}

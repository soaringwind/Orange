// #pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <stack>
#include <set>
#include <queue>
#include "dataFrame.cpp"

template <class T1, class T2>
double Pearson(std::vector<T1> &_in1, std::vector<T2> &_in2)
{
	if (_in1.size() != _in2.size())
	{
		std::cout << "尺寸不匹配" << std::endl;
		return 0.f;
	}
	size_t n = _in1.size();
	double pearson = n * std::inner_product(_in1.begin(), _in1.end(), _in2.begin(), 0.f) - std::accumulate(_in1.begin(), _in1.end(), 0.f) * std::accumulate(_in2.begin(), _in2.end(), 0.f);
	double tem1 = n * std::inner_product(_in1.begin(), _in1.end(), _in1.begin(), 0.f) - pow(std::accumulate(_in1.begin(), _in1.end(), 0.f), 2);
	double tem2 = n * std::inner_product(_in2.begin(), _in2.end(), _in2.begin(), 0.f) - pow(std::accumulate(_in2.begin(), _in2.end(), 0.f), 2);
	tem1 = sqrt(tem1);
	tem2 = sqrt(tem2);
	pearson = pearson / (tem1 * tem2);
	return pearson;
}

template <class T1, class T2>
double Spearman(std::vector<T1> &_in1, std::vector<T2> &_in2)
{
	if (_in1.size() != _in2.size())
	{
		std::cout << "尺寸不匹配" << std::endl;
		return 0.f;
	}
	size_t n = _in1.size();
	double spearman{0};
	std::vector<int> _in1Index;
	std::vector<int> _in2Index;
	for (int i = 0; i < n; i++)
	{
		_in1Index.emplace_back(i);
		_in2Index.emplace_back(i);
	}
	std::sort(_in1Index.begin(), _in1Index.end(), [_in1, _in1Index](int a, int b)
			  { return _in1[_in1Index[a]] < _in1[_in1Index[b]]; });
	std::sort(_in2Index.begin(), _in2Index.end(), [_in2, _in2Index](int a, int b)
			  { return _in2[_in2Index[a]] < _in2[_in2Index[b]]; });
	for (int i = 0; i < n; i++)
	{
		spearman += pow((_in1Index[i] - _in2Index[i]), 2);
	}
	spearman = 1 - (6 * spearman / (n * (pow(n, 2) - 1)));
	return spearman;
}
template <class T>
float calcDistance(std::vector<T> data1, std::vector<T> data2)
{
	float tem = 0.f;
	for (int i = 0; i < data1.size(); ++i)
	{
		tem += pow((data1[i] - data2[i]), 2);
	}
	return sqrt(tem);
}

template <class T>
std::vector<int> DBscan(std::vector<std::vector<T>> &data, float eps, int minPts, float (*func)(std::vector<T>, std::vector<T>) = calcDistance)
{
	// 以下实现为brute算法时间效率k*n**2 (如果传入的不是可以直接用欧式距离衡量的统一进行brute算法)

	// 数据的数量
	int n = data.size();

	// 返回的分类结果
	std::vector<int> clusters = std::vector<int>(n, -1);

	// 是否已经被访问
	std::vector<bool> visited = std::vector<bool>(n);

	// 每个点的类型 (1: noise, 2: border, 3: Core)
	std::vector<int> pointType = std::vector<int>(n);

	// 核心点的索引
	std::vector<int> corePoint = std::vector<int>();

	// 每个数据与别的数据的距离
	std::unordered_map<int, std::vector<float>> distanceMap = std::unordered_map<int, std::vector<float>>();

	// 确定核心点并计算每个点相互之间的距离, 复杂度实际上是k*n**2, 空间换时间
	for (int i = 0; i < n; ++i)
	{
		int num = 0;
		std::vector<float> perDis;
		for (int j = 0; j < n; ++j)
		{
			float res = func(data[i], data[j]);
			if (res < eps)
			{
				num += 1;
			}
			perDis.emplace_back(res);
		}
		distanceMap[i] = perDis;
		if (num > minPts)
		{
			pointType[i] = 3;
			corePoint.emplace_back(i);
		}
		else
		{
			pointType[i] = 1;
		}
	}

	// 打乱索引
	srand(n);
	std::random_shuffle(corePoint.begin(), corePoint.end());

	// 核心点扩展
	std::stack<int> ps;
	int core;
	int clusterNum = -1;
	for (int i = 0; i < corePoint.size(); ++i)
	{
		core = corePoint[i];
		if (visited[core])
			continue;
		ps.push(core);
		++clusterNum;
		clusters[core] = clusterNum;
		while (!ps.empty())
		{
			core = ps.top();
			ps.pop();
			visited[core] = true;
			auto coreDis = distanceMap[core];
			// 搜索距离核心点满足条件的点数
			for (int j = 0; j < coreDis.size(); ++j)
			{
				if (visited[j] || coreDis[j] > eps)
					continue;
				if (pointType[j] == 3)
				{
					ps.push(j);
					clusters[j] = clusterNum;
				}
				else
				{
					clusters[j] = clusterNum;
					visited[j] = true;
				}
			}
		}
	}
	return clusters;
}
struct KdNode
{
	// 构建kdtree的node节点
	int id = 0;
	int splitDim = 0;
	KdNode *left = nullptr;
	KdNode *right = nullptr;
};
class KdTree
{
public:
	// 根节点
	KdNode *root{};

	// 保存输入数据
	std::vector<std::vector<double>> data{};

	// 初始化的数据索引
	std::vector<int> index{};

	// 返回预测数据的标签即接近数据的索引
	std::vector<int> label{};

	// 数据行数
	int n_samples{};

	// 数据特征数
	int n_features{};

	KdTree(std::vector<std::vector<double>> &data);

	// 建树
	KdNode *buildTree(std::vector<int> &ptIndex, int depth = 0);

	// 找到k个最近邻
	void findKNearnests(const std::vector<double> &perdictData, int &k);

	// 找到分裂的维度
	int findSplitDim(const std::vector<int> &ptIndex);

	// 查看整颗树结构
	void printTree();

	~KdTree();

private:
};

KdTree::KdTree(std::vector<std::vector<double>> &data)
{
	this->data = data;
	this->n_samples = this->data.size();
	this->n_features = this->data[0].size();
	for (int i = 0; i < n_samples; ++i)
	{
		this->index.emplace_back(i);
	}
	this->root = this->buildTree(this->index);
}
inline KdNode *KdTree::buildTree(std::vector<int> &ptIndex, int depth)
{
	if (ptIndex.empty())
		return nullptr;

	// 选择差值最大的特征选择切割维度
	int dim = this->findSplitDim(ptIndex);

	// 根据深度选择切割维度
	// int dim = depth % n_features;

	// 选择给定维度中间的数据进行左右分割
	std::nth_element(ptIndex.begin(), ptIndex.begin() + ptIndex.size() / 2, ptIndex.end(), [dim, this](int &i, int &j)
					 { return data[i][dim] < data[j][dim]; });
	std::vector<int> left(ptIndex.begin(), ptIndex.begin() + ptIndex.size() / 2);
	std::vector<int> right(ptIndex.begin() + 1 + (ptIndex.size() / 2), ptIndex.end());
	KdNode *node = new KdNode{};
	node->id = *(ptIndex.begin() + ptIndex.size() / 2);
	node->splitDim = dim;
	node->left = this->buildTree(left, depth + 1);
	node->right = this->buildTree(right, depth + 1);
	return node;
}

inline int KdTree::findSplitDim(const std::vector<int> &ptIndex)
{
	int dim = 0;
	double difference = 0;
	for (int i = 0; i < this->n_features; ++i)
	{
		double min_val = this->data[ptIndex[0]][i];
		double max_val = this->data[ptIndex[0]][i];
		for (int j = 0; j < ptIndex.size(); ++j)
		{
			if (this->data[ptIndex[j]][i] <= min_val)
			{
				min_val = this->data[ptIndex[j]][i];
			}
			if (this->data[ptIndex[j]][i] >= max_val)
			{
				max_val = this->data[ptIndex[j]][i];
			}
		}
		if ((max_val - min_val) >= difference)
		{
			dim = i;
			difference = max_val - min_val;
		}
	}
	return dim;
}
void KdTree::findKNearnests(const std::vector<double> &perdictData, int &k)
{
	// 自定义比较元素
	struct neighbour_heap_cmp
	{
		bool operator()(std::tuple<int, double> &i, std::tuple<int, double> &j)
		{
			return std::get<1>(i) < std::get<1>(j);
		}
	};

	// 组织的数据结构, <index, distance>
	using neighbour = std::tuple<int, double>;
	KdNode *currentNode = this->root;
	std::set<KdNode *> visited;

	// 大顶堆存放原始数据索引及其与预测点的距离
	std::priority_queue<neighbour, std::vector<neighbour>, neighbour_heap_cmp> neighbour_heap;

	// 存放KdNode的搜索路径
	std::stack<KdNode *> searchPath;
	searchPath.push(currentNode);
	while (currentNode->left != nullptr && currentNode->right != nullptr)
	{
		if (currentNode->right == nullptr || perdictData[currentNode->splitDim] < this->data[currentNode->id][currentNode->splitDim])
		{
			currentNode = currentNode->left;
		}
		else
		{
			currentNode = currentNode->right;
		}
		searchPath.push(currentNode);
	}

	// 搜索路径不为空则一直查找
	while (!searchPath.empty())
	{
		currentNode = searchPath.top();
		searchPath.pop();
		if (currentNode == nullptr || visited.find(currentNode) != visited.end())
		{
			continue;
		}
		visited.insert(currentNode);
		if (neighbour_heap.size() < k)
		{
			neighbour_heap.emplace(std::tuple<int, double>(currentNode->id, double(calcDistance(this->data[currentNode->id], perdictData))));
			searchPath.push(currentNode->left);
			searchPath.push(currentNode->right);
		}
		else
		{
			double p = std::get<1>(neighbour_heap.top());
			if (p > calcDistance(this->data[currentNode->id], perdictData))
			{
				neighbour_heap.pop();
				neighbour_heap.emplace(std::tuple<int, double>(currentNode->id, double(calcDistance(this->data[currentNode->id], perdictData))));
			}
			p = std::get<1>(neighbour_heap.top());
			int dim = currentNode->splitDim;
			if ((perdictData[dim] - p) < this->data[currentNode->id][dim])
			{
				searchPath.push(currentNode->left);
			}
			if ((perdictData[dim] + p) > this->data[currentNode->id][dim])
			{
				searchPath.push(currentNode->right);
			}
		}
	}

	// 将结果保存至label标签中
	for (int i = 0; i < k; i++)
	{
		this->label.insert(this->label.begin(), std::get<0>(neighbour_heap.top()));
		neighbour_heap.pop();
	}
	return;
}
inline void KdTree::printTree()
{
	std::queue<KdNode *> tem;
	tem.push(this->root);
	while (!tem.empty())
	{
		int n = tem.size();
		for (int i = 0; i < n; ++i)
		{
			auto node = tem.front();
			std::cout << "(" << node->splitDim << ", " << node->id << ")"
					  << " ";
			tem.pop();
			if (node->left)
			{
				tem.emplace(node->left);
			}
			if (node->right)
			{
				tem.emplace(node->right);
			}
		}
		std::cout << "\n";
	}
}
KdTree::~KdTree()
{
}
void knn()
{
}
void testDBscan()
{
	std::vector<float> vec1;
	std::vector<float> vec2;
	vec1.emplace_back(1.f);
	vec1.emplace_back(2.f);
	vec1.emplace_back(3.f);
	vec1.emplace_back(2.f);
	vec2.emplace_back(1.f);
	vec2.emplace_back(0.f);
	vec2.emplace_back(4.f);
	vec2.emplace_back(2.f);
	auto vec3 = std::vector<std::vector<float>>();
	vec3.emplace_back(vec1);
	vec3.emplace_back(vec2);
	DBscan(vec3, 0.3, 1);
}
void testPearson()
{
	std::vector<float> vec1;
	std::vector<float> vec2;
	vec1.emplace_back(1.f);
	vec1.emplace_back(2.f);
	vec1.emplace_back(3.f);
	vec1.emplace_back(2.f);
	vec2.emplace_back(1.f);
	vec2.emplace_back(0.f);
	vec2.emplace_back(4.f);
	vec2.emplace_back(2.f);
	std::cout << Pearson(vec1, vec2) << std::endl;
	std::cout << Spearman(vec1, vec2) << std::endl;
}
void testDataDbscan()
{
	DataFrame df{};
	df.read_csv(R"(./data.csv)");
	std::cout << "columns: " << df.column_length << " "
			  << "rows: " << df.row_length << std::endl;
	auto res = df.values<double>();
	std::cout.precision(16);
	std::cout << res[0][0] << std::endl;
	auto cluster = DBscan(res, 5, 3);
	for (int i = 0; i < cluster.size(); ++i)
	{
		std::cout << cluster[i] << std::endl;
	}
}
void testKnn()
{
	DataFrame df{};
	df.read_csv(R"(./data.csv)");
	std::cout << "columns: " << df.column_length << " "
			  << "rows: " << df.row_length << std::endl;
	auto res = df.values<double>();
	auto knn = KdTree(res);
	std::vector<double> perdict;
	perdict.emplace_back(7.723325119394918);
	perdict.emplace_back(0.7898756767401094); // n=40
	int k = 3;
	knn.findKNearnests(perdict, k);
	for (int i = 0; i < knn.label.size(); i++)
	{
		std::cout << knn.label[i] << std::endl;
	}
	// knn.printTree();
}
int main()
{
	testKnn();
	return 0;
}

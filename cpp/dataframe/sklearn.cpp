// #pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <stack>
#include "dataFrame.cpp"

template <class T1, class T2>
double Pearson(std::vector<T1>& _in1, std::vector<T2>& _in2)
{
	if (_in1.size() != _in2.size())
	{
		std::cout << "�ߴ粻ƥ��" << std::endl;
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
double Spearman(std::vector<T1>& _in1, std::vector<T2>& _in2)
{
	if (_in1.size() != _in2.size())
	{
		std::cout << "�ߴ粻ƥ��" << std::endl;
		return 0.f;
	}
	size_t n = _in1.size();
	double spearman{ 0 };
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
std::vector<int> DBscan(std::vector<std::vector<T>>& data, float eps, int minPts, float(*func)(std::vector<T>, std::vector<T>) = calcDistance)
{
	// ����ʵ��Ϊbrute�㷨ʱ��Ч��k*n**2 (�������Ĳ��ǿ���ֱ����ŷʽ���������ͳһ����brute�㷨)

	// ���ݵ�����
	int n = data.size();

	// ���صķ�����
	std::vector<int> clusters = std::vector<int>(n, -1);

	// �Ƿ��Ѿ�������
	std::vector<bool> visited = std::vector<bool>(n);

	// ÿ��������� (1: noise, 2: border, 3: Core)
	std::vector<int> pointType = std::vector<int>(n);

	// ���ĵ������
	std::vector<int> corePoint = std::vector<int>();

	// ÿ�������������ݵľ���
	std::unordered_map<int, std::vector<float>> distanceMap = std::unordered_map<int, std::vector<float>>();

	// ȷ�����ĵ㲢����ÿ�����໥֮��ľ���, ���Ӷ�ʵ������k*n**2, �ռ任ʱ��
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

	// ��������
	srand(n);
	std::random_shuffle(corePoint.begin(), corePoint.end());

	// ���ĵ���չ
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
			// ����������ĵ����������ĵ���
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
struct KdNode {
	// ����kdtree��node�ڵ�
	int id;
	int splitDim;
	KdNode* left=nullptr;
	KdNode* right=nullptr;
};
class KdTree
{
public:
	// ���ڵ�
	KdNode* root;

	// ������������
	std::vector<std::vector<double>> data;

	// ��ʼ������������
	std::vector<int> index;

	// ����Ԥ�����ݵı�ǩ���ӽ����ݵ�����
	int label;

	// ��������
	int n_samples;

	// ����������
	int n_features;

	KdTree();

	// ����
	KdNode* buildTree(std::vector<int>& ptIndex);

	// �ҵ�k�������
	void findKNearnests(const std::vector<double>& perdictData);

	// �ҵ����ѵ�ά��
	int findSplitDim(const std::vector<int>& ptIndex);

	~KdTree();

private:

};

KdTree::KdTree()
{
}

inline KdNode* KdTree::buildTree(std::vector<int>& ptIndex)
{
	if (ptIndex.empty()) return nullptr;
	int dim = this->findSplitDim(ptIndex);
	std::nth_element(ptIndex.begin(), ptIndex.begin() + ptIndex.size() / 2, ptIndex.end(), [dim, this](int& i, int& j) {return data[i][dim] < data[j][dim]; });
	std::vector<int> left(ptIndex.begin(), ptIndex.begin() + ptIndex.size() / 2);
	std::vector<int> right(ptIndex.begin() + 1 + (ptIndex.size() / 2), ptIndex.end());
	KdNode* node{};
	node->id = *(ptIndex.begin() + ptIndex.size() / 2);
	node->splitDim = dim;
	node->left = this->buildTree(left);
	node->right = this->buildTree(right);
	return node;
}

inline int KdTree::findSplitDim(const std::vector<int>& ptIndex)
{
	return 0;
}

KdTree::~KdTree()
{
}
void knn() {

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
void testDataDbscan() {
	DataFrame df{};
	df.read_csv(R"(./data.csv)");
	std::cout << "��: " << df.column_length << " " << "��: " << df.row_length << std::endl;
	auto res = df.values<double>();
	std::cout.precision(16);
	std::cout << res[0][0] << std::endl;
	auto cluster = DBscan(res, 5, 3);
	for (int i = 0; i < cluster.size(); ++i) {
		std::cout << cluster[i] << std::endl;
	}
}
int main()
{
	testDataDbscan();
	return 0;
}

// #pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <algorithm>

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
	double tem1 = n * std::inner_product(_in1.begin(), _in1.end(), _in1.begin(), 0.f) - std::pow(std::accumulate(_in1.begin(), _in1.end(), 0.f), 2);
	double tem2 = n * std::inner_product(_in2.begin(), _in2.end(), _in2.begin(), 0.f) - std::pow(std::accumulate(_in2.begin(), _in2.end(), 0.f), 2);
	tem1 = std::sqrt(tem1);
	tem2 = std::sqrt(tem2);
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
	double spearman;
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

int main()
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
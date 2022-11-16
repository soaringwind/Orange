#pragma once
#include <vector>
#include <numeric>
#include <iostream>


template<class T1, class T2>
double Person(std::vector<T1>& _in1, std::vector<T2>& _in2) {
	if (_in1.size() != _in2.size()) {
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

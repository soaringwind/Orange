#pragma once
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <sstream>
#include <numeric>
#include <fstream>
#include <iostream>
#include <type_traits>


enum ObjectId
{
	intergId = 1,
	floatId = 2,
	doubleId = 3,
	longId = 4,
	stringId = 5,
};


class Cell
{
public:
	// 储存cell的值指针
	void* value;

	// 储存cell的值类型
	ObjectId dtype;

	// 将cell值转成字符串
	std::string to_string();

	// 修改cell值，传入的值可能有多种数据类型
	template<class T>
	void update_value(T& _in);

	// 修改cell数据类型
	void update_dtype(int);
	void update_dtype(float);

	// 深拷贝构造函数
	Cell copy();
};

class Series
{

};

class DataFrame
{
public:
	// 储存行数
	int row_length = 0;

	// 储存列数
	int column_length = 0;

	// 储存列名
	std::vector<std::string> columnList;

	// 储存列数据类型
	std::vector<int> dtypeList;

	// 储存数据
	std::vector<std::vector<Cell*>> data;

	// 构造函数
	DataFrame();

	// 插入列，列可能有多种数据结构
	void insert(std::string&& columnName, std::vector<int>& data);

	// 根据列名查找列
	std::vector<std::string> find(std::string&& columnName);

	// 根据列名取出对应列
	Series get(std::string&& columnName);

	// 输出到csv文件
	void to_csv(std::string&& keyName);

	// 深拷贝函数
	DataFrame copy();

	// 析构函数
	~DataFrame();
private:
};


template<class T>
inline void Cell::update_value(T& _in)
{
	if (std::is_same<int, T>::value)
	{
		this->value = &_in;
		this->dtype = intergId;
	}
	else if (std::is_same<float, T>::value)
	{
		this->value = &_in;
		this->dtype = floatId;
	}
	else if (std::is_same<double, T>::value)
	{
		this->value = &_in;
		this->dtype = doubleId;
	}
	else if (std::is_same<long, T>::value)
	{
		this->value = &_in;
		this->dtype = longId;
	}
	else if (std::is_same<std::string, T>::value)
	{
		this->value = &_in;
		this->dtype = stringId;
	}
	else
	{
		return;
	}
	return;
}




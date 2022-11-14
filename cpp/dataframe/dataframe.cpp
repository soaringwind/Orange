// #pragma once
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
	void *value;

	// 储存cell的值类型
	ObjectId dtype;

	// 将cell值转成字符串
	std::string to_string();

	// 修改cell值，传入的值可能有多种数据类型
	template <class T>
	void update_value(T &_in);

	// 修改cell数据类型
	void update_dtype(int objId);

	// 深拷贝构造函数
	Cell copy();
};

class Series
{
public:
	std::string name;
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
	std::vector<std::vector<Cell *>> data;

	// 构造函数
	DataFrame();

	// 插入列，列可能有多种数据结构
	void insert(std::string &&columnName, std::vector<int> &data);

	// 根据列名查找列
	std::vector<std::string> find(std::string &&columnName);

	// 根据列名取出对应列
	Series get(std::string &&columnName);

	// 输出到csv文件
	void to_csv(std::string &&keyName);

	// 深拷贝函数
	DataFrame copy();

	// 析构函数
	~DataFrame();

private:
};

template <class T>
inline void Cell::update_value(T &_in)
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

void Cell::update_dtype(int objId)
{
	std::string tem = this->to_string();
	switch (objId)
	{
	case intergId:
	{
		int tem_i = std::stoi(tem);
		this->update_value(tem_i);
		break;
	}
	case floatId:
	{
		float tem_f = std::stof(tem);
		this->update_value(tem_f);
		break;
	}
	case doubleId:
	{
		double tem_d = std::stod(tem);
		this->update_value(tem_d);
		break;
	}
	case longId:
	{
		long tem_l = std::stol(tem);
		this->update_value(tem_l);
		break;
	}
	case stringId:
		this->update_value(tem);
		break;
	default:
		this->update_value(tem);
		break;
	}
}

std::string Cell::to_string()
{
	switch (this->dtype)
	{
	case intergId:
		return std::to_string(*(int *)this->value);
	case floatId:
		return std::to_string(*(float *)this->value);
	case doubleId:
		return std::to_string(*(double *)this->value);
	case longId:
		return std::to_string(*(long *)this->value);
	case stringId:
		return *(std::string *)this->value;
	default:
		break;
	}
	return std::string();
}
Cell Cell::copy() {
	Cell* newCell = new Cell{};
	switch (this->dtype)
	{
	case intergId:
	{
		int tem_i = *(int*)value;
		newCell->value = &tem_i;
		newCell->dtype = intergId;
		break;
	}
	case floatId:
	{
		float tem_f = *(float*)value;
		newCell->value = &tem_f;
		newCell->dtype = floatId;
		break;
	}
	case doubleId:
	{
		double tem_d = *(double*)value;
		newCell->value = &tem_d;
		newCell->dtype = doubleId;
		break;
	}
	case longId:
	{
		long tem_l = *(long*)value;
		newCell->value = &tem_l;
		newCell->dtype = longId;
		break;
	}
	case stringId:
	{
		std::string tem_s = *(std::string *)value;
		newCell->value = &tem_s;
		newCell->dtype = stringId;
		break;
	}
	default:
		break;
	}
	return *newCell;
}

int main()
{
	Cell cell1{};
	int a = 1;
	cell1.value = &a;
	cell1.dtype = intergId;
	int b = 2;
	cell1.update_value(b);
	std::cout << cell1.to_string() << std::endl;
	auto cell2 = cell1.copy();
	cell2.update_value(a);
	std::cout << cell2.to_string() << std::endl;
	std::cout << cell1.to_string() << std::endl;
}

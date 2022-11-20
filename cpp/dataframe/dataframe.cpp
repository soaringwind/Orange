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
#include <fstream>

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

	// 得到cell的值
	template <class T>
	void getCell(T &_in);

	// 深拷贝构造函数
	Cell copy();
};

class Series
{
public:
	// Series名
	std::string name;
};

class DataFrame
{
public:
	// 储存行数
	int row_length = 0;

	// 储存列数
	int column_length = 0;

	// 储存列名(这里应该考虑map/multi_map)
	std::vector<std::string> columnList;

	// 储存列数据类型
	std::vector<int> dtypeList;

	// 储存数据
	std::vector<std::vector<Cell *>> data;

	// 构造函数
	DataFrame();

	// 插入列，列可能有多种数据结构
	template <class T>
	void insert(std::string &&columnName, std::vector<T> &data);

	// 根据列名查找列
	std::vector<std::string> findCol(std::string &&columnName);

	// 查找列索引
	int findColIndex(std::string &&columnName);

	// 拿到df数据矩阵
	template <class T>
	std::vector<std::vector<T>> values();

	// 根据列名取出对应列
	Series get(std::string &&columnName);

	// 根据列名取对应列数据
	template <class T>
	std::vector<T> getColValue(std::string &&columnName);

	// 输出到csv文件
	void to_csv(std::string &&fileName);

	// 从csv中读取数据
	void read_csv(std::string &&fileName);

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
template <class T>
void Cell::getCell(T &_in)
{
	_in = *(T *)this->value;
}
Cell Cell::copy()
{
	Cell newCell{};
	switch (this->dtype)
	{
	case intergId:
	{
		int tem_i = *(int *)value;
		newCell.value = &tem_i;
		newCell.dtype = intergId;
		break;
	}
	case floatId:
	{
		float tem_f = *(float *)value;
		newCell.value = &tem_f;
		newCell.dtype = floatId;
		break;
	}
	case doubleId:
	{
		double tem_d = *(double *)value;
		newCell.value = &tem_d;
		newCell.dtype = doubleId;
		break;
	}
	case longId:
	{
		long tem_l = *(long *)value;
		newCell.value = &tem_l;
		newCell.dtype = longId;
		break;
	}
	case stringId:
	{
		std::string tem_s = *(std::string *)value;
		newCell.value = &tem_s;
		newCell.dtype = stringId;
		break;
	}
	default:
		break;
	}
	return newCell;
}
inline DataFrame::DataFrame()
{
}
template <class T>
void DataFrame::insert(std::string &&columnName, std::vector<T> &data)
{
	ObjectId Id = intergId;
	if (std::is_same<int, T>::value)
	{
		Id = intergId;
	}
	else if (std::is_same<float, T>::value)
	{
		Id = floatId;
	}
	else if (std::is_same<double, T>::value)
	{
		Id = doubleId;
	}
	else if (std::is_same<long, T>::value)
	{
		Id = longId;
	}
	else if (std::is_same<std::string, T>::value)
	{
		Id = stringId;
	}
	else
	{
		std::cout << "无类型" << std::endl;
		return;
	}
	for (int i = 0; i < data.size(); i++)
	{
		// 如果不使用new，则开辟在栈区
		Cell *newCell = new Cell{};
		newCell->value = &data[i];
		newCell->dtype = Id;
		auto newVec = new std::vector<Cell *>;
		if (this->column_length != 0)
		{
			newVec = &this->data[i];
		}
		newVec->emplace_back(newCell);
		if (this->column_length == 0)
		{
			this->data.emplace_back(*newVec);
		}
	}
	this->columnList.emplace_back(columnName);
	this->column_length += 1;
	this->dtypeList.emplace_back(Id);
	this->row_length = std::max(row_length, int(data.size()));
}
std::vector<std::string> DataFrame::findCol(std::string &&columnName)
{
	std::vector<std::string> res;
	return res;
}
int DataFrame::findColIndex(std::string &&columnName)
{
	int res = -1;
	for (int i = 0; i < this->column_length; i++)
	{
		if (this->columnList[i] == columnName)
		{
			res = i;
			break;
		}
	}
	return res;
}
template <class T>
std::vector<std::vector<T>> DataFrame::values()
{
	std::vector<std::vector<T>> res;
	for (int i = 0; i < this->row_length - 1; ++i)
	{
		std::vector<T> *tem = new std::vector<T>;
		for (int j = 0; j < this->column_length; ++j)
		{
			tem->emplace_back(*(T *)this->data[i][j]->value);
		}
		res.emplace_back(*tem);
	}
	return res;
}
template <class T>
std::vector<T> DataFrame::getColValue(std::string &&columnName)
{
	std::vector<T> res;
	auto colIndex = this->findColIndex(std::move(columnName));
	if (colIndex != -1) {
		for (int i=0; i<this->row_length-1; i++) {
			res.emplace_back(*(T*)this->data[i][colIndex]->value);
		}
	}
	return res;
}
inline void DataFrame::to_csv(std::string &&fileName)
{
	// 暂且写在一起，且不做多线程优化
	std::ofstream outFile;
	std::stringstream ss;
	outFile.open(fileName, std::ios::out | std::ios::trunc);
	if (!outFile.good())
	{
		std::cout << "打开文件失败" << std::endl;
		return;
	}
	for (int i = 0; i < this->column_length; i++)
	{
		ss << this->columnList[i];
		if (i != this->column_length - 1)
		{
			ss << ", ";
		}
		else
		{
			ss << "\n";
		}
	}
	for (int i = 0; i < this->data.size(); i++)
	{
		for (int j = 0; j < this->data[i].size(); j++)
		{
			ss << this->data[i][j]->to_string();
			if (j != this->data[i].size() - 1)
			{
				ss << ", ";
			}
		}
		ss << "\n";
	}
	outFile << ss.str();
	outFile.close();
}
void DataFrame::read_csv(std::string &&fileName)
{
	// 目前实现的是读入double, 计划的实现是能够读入double/string两种格式
	std::ifstream inFile;
	std::string token;
	inFile.open(fileName);
	while (std::getline(inFile, token))
	{
		++this->row_length;
		std::istringstream iss(token);
		std::vector<Cell *> *temData = new std::vector<Cell *>;
		while (std::getline(iss, token, ','))
		{
			if (this->row_length <= 1)
			{
				++this->column_length;
				this->columnList.emplace_back(token);
				this->dtypeList.emplace_back(doubleId);
				continue;
			}
			Cell *newCell = new Cell{};
			double *tem = new double;
			*tem = std::stod(token);
			newCell->value = tem;
			newCell->dtype = doubleId;
			temData->emplace_back(newCell);
		}
		if (this->row_length > 1)
			this->data.emplace_back(*temData);
	}
}

inline DataFrame::~DataFrame()
{
}
void testCell()
{
	Cell cell1;
	int a = 1;
	cell1.value = &a;
	cell1.dtype = intergId;
	int b;
	cell1.getCell(b);
	std::cout << b << std::endl;
}
void testUseDf()
{
	DataFrame df{};
	std::vector<int> vec;
	vec.emplace_back(1);
	vec.emplace_back(2);
	vec.emplace_back(1);
	std::vector<int> vec1;
	vec1.emplace_back(1);
	vec1.emplace_back(2);
	vec1.emplace_back(1);
	df.insert("first", vec);
	df.insert("second", vec1);
	df.to_csv("testdf.csv");
}
void testReadDf()
{
	DataFrame df{};
	df.read_csv(R"(C:\Users\weitao\Desktop\Untitled Folder\cpp\sklearn\data.csv)");
	std::cout << df.column_length << " " << df.row_length << std::endl;
	std::cout << df.data[0][0]->to_string() << std::endl;
	auto res = df.values<double>();
	std::cout.precision(16);
	std::cout << res[0][0] << std::endl;
	std::cout << df.findColIndex("0") << std::endl;
	auto colRes = df.getColValue<double>("0");
	std::cout << colRes[0] << std::endl;
}
int main()
{
	testReadDf();
}

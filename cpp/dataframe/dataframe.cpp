#include <vector>
#include <string>
#include <map>
#include <iostream>


class BaseColumn
{};


template<class T>
class Column:public BaseColumn
{
public:
    Column(std::string &colName, std::vector<T> &data)
    {
        this->m_column = colName;
        this->m_vec = data;
    }
    std::string m_column;
    std::vector<T> m_vec;
};

class MyDataFrame
{
public:
    std::vector<std::string> m_columnVec;
    std::vector<BaseColumn*> m_dataVec;
    template<class T>
    void insertColumn(std::string colName, std::vector<T> data)
    {
        Column<T>* col = new Column<T>(colName, data);
        this->m_dataVec.emplace_back(col);
        this->m_columnVec.emplace_back(colName);
    }
    template<class T>
    std::vector<T> getColumn(std::string colName)
    {
        for (int i=0; i<this->m_columnVec.size(); i++)
        {
            if (this->m_columnVec[i] == colName) 
            {
                return this->m_dataVec[i];
            }
        }
    }
};

int main(){
    MyDataFrame df;
    std::string colName = "index";
    std::vector<int> data;
    for (int i=0; i< 10; i++) {
        data.emplace_back(i);
    }
    df.insertColumn<int>(colName, data);
    auto res = df.getColumn<int>(colName);
    for (int i=0; i<res.size(); i++) {
        std::cout << i << std::endl;
    }
}
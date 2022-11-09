#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
using namespace std;

class Base
{
public:
    static int a;
};
int Base::a;

template <class T>
class Value : public Base
{
public:
    double m_key;
    T m_v;
};

class Column
{
public:
    template <class T>
    static T data;
};
template<class T>
T Column::data;


template<class T>
class Unit
{
public:
    T data;
    T getData()
    {
        return data;
    }
};

class Cell
{
public:
    template <class T>
    static T data;
    string dtype;
    Cell(int&& val)
    {
        this->data<int> = val;
        this->dtype = "int";
    }
    Cell(float&& val)
    {
        this->data<float> = val;
        this->dtype = "float";
    }
    Cell(string&& val)
    {
        this->data<string> = val;
        this->dtype = "string";
    }
    string getValue()
    {
        if (this->dtype=="string")
        {
            return this->data<string>;
        }
        if (this->dtype=="int")
        {
            return to_string(this->data<int>);
        }
        if (this->dtype=="float")
        {
            return to_string(this->data<float>);
        }
        return string(" ");
    }
};
template<class T>
T Cell::data;

int main()
{
    Cell cell1(1);
    Cell cell2{2.f};
    Cell cell3{"111"};
    cout << cell1.getValue() << cell2.getValue() << cell3.getValue() << endl;
    Unit<int> unit1;
    unit1.data = 1;
    cout << unit1.getData() << endl;
    // vector<Base*> vec;
    // Value<int>* v1 = new Value<int>;
    // v1->m_key = 0.0;
    // vec.emplace_back(v1);
    // Base base1;
    // base1.a = 1;
    // Base base2;
    // base2.a = 2;
    // cout << base1.a << endl;
    // ifstream inFile;
    // inFile.open("./test.txt", ios::in);
    // char str1[4];
    // inFile.read(str1, sizeof(str1));
    // inFile.getline(str1, sizeof(str1));
    // cout << str1 << endl;
}
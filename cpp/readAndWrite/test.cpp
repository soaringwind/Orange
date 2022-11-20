#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <ctime>
#include <regex>
#include <numeric>
#include <limits>
#include <stdio.h>


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



int func(int i) {
    cout << i << ", ";
    return 0;
}

void stringSplit(string str, const char* delim) {
    ifstream inFile;
    inFile.open(R"(C:\Users\weitao\Desktop\Untitled Folder\cpp\sklearn\data.csv)");
    string token;
    while (getline(inFile, token)) {
        istringstream iss(token);
        while (getline(iss, token, *delim))
        {                                                                                                                                                                                                          
            /* code */
            cout << token << endl;
        }
        break;
    }

    
}

int main()
{
    // multimap<char, int>mymultimap{ {'a',10},{'b',20},{'b',15}, {'c',30} };
    // cout << mymultimap.find('b')->second << endl;

    // vector<int> vec;
    // for (int i=0; i<10; i++) {
    //     vec.emplace_back(i);
    // }
    // for_each(vec.begin(), vec.end(), func);
    // srand(12);
    // random_shuffle(vec.begin(), vec.end());
    // cout << "\n";
    // for_each(vec.begin(), vec.end(), func);

    // string text = "Quick brown fox.";
    // regex ws_re("\\s+");
    // vector<string> v(sregex_token_iterator(text.begin(), text.end(), ws_re, -1), sregex_token_iterator());
    // for (auto&& s: v) cout << s << "\n";

    // string str("this, is, a test");
    // stringSplit(str, ",");

    // vector<int>* a = new vector<int>;
    // a->emplace_back(1);
    // cout << (*a)[0] << endl;

    int prec = numeric_limits<double>::digits10;
    double tem;
    string value = "1.124124214124";
    tem = std::stod(value);
    stringstream ss;
    // ss << value;
    // cout << ss.str() << endl;
    // ss >> tem;
    // cout << tem << endl;
    // ss.clear();
    // ss.str("");
    // ss.precision(value.length());
    // ss << tem;
    // cout << ss.str() << endl;
    // cout.precision(16);
    // cout << tem << endl;
    ss.precision(16);
    ss << tem;
    cout.precision(16);
    cout << ss.str() << endl;

    // cout << stoi(" 100") << endl;
    // vector<int> vec;
    // vec.emplace_back(func());
    // cout <<  &vec[0] << endl;
    // Cell cell1(1);
    // Cell cell2{2.f};
    // Cell cell3{"111"};
    // cout << cell1.getValue() << cell2.getValue() << cell3.getValue() << endl;
    // Unit<int> unit1;
    // unit1.data = 1;
    // cout << unit1.getData() << endl;
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
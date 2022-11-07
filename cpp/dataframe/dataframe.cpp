#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace std;

class MyDataFrame
{
public:
    string colName;
    template<class T>
    static vector<T> data;
private:
};

template<class T>
T pi = T(3.14);
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <unordered_map>


using namespace std;



// class Contain
// {
// public:
//     template<class T>
//     static unordered_map<Contain*, vector<T>> data;
//     template<class T>
//     void insert(T& _in)
//     {
//         data<T>[this].push_back(_in);
//     }
//     template<class T>
//     void visit(T&& visitor)
//     {
//         visitor();
//     }
// };


// template<class T>
// unordered_map<Contain*, vector<T>> Contain::data;


template<class T>
class Unit
{
public:
    T value;
    T getValue() {
        return value;
    }
};


class Cell
{
public:
    void* obj;
    string str_val;
};

int main(){
    Unit<int> unit1;
    unit1.value = 1;
    Cell cell1;
    cell1.obj = &unit1;
    cell1.str_val = to_string(unit1.value);
    auto pt = *(Unit<int>*)cell1.obj;
    cout << pt.getValue() << endl;
    cout << cell1.str_val << endl;
}

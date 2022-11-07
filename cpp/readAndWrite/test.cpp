#include <iostream>
#include <fstream>
#include <vector>

class Base
{};

template <class T>
class Value:public Base
{
    public:
    double m_key;
    T m_v;
};


using namespace std;
int main() {
    vector<Base*> vec;
    Value<int>* v1 = new Value<int>;
    v1->m_key = 0.0;
    vec.emplace_back(v1);
    // ifstream inFile;
    // inFile.open("./test.txt", ios::in);
    // char str1[4];
    // inFile.read(str1, sizeof(str1));
    // inFile.getline(str1, sizeof(str1));
    // cout << str1 << endl;
}
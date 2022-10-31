#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>
#include <map>
#include "boost/version.hpp"

using namespace std;


class hello
{
private:
   /* data */
public:
   hello(/* args */);
   ~hello();
};

hello::hello(/* args */)
{
}

hello::~hello()
{
}


int main(void)
{
   // cout << BOOST_VERSION << endl;
   // vector<int> v1;
   // v1.emplace_back(1);
   // v1.emplace_back(1);
   // vector<int>::iterator it = v1.end();
   // cout << *it << endl;
   // deque<int> d1;
   map<string, int> m1;
   hello p1;
   m1.insert(pair<string, int>("1", 1));
   m1.insert(make_pair("2", 2));
   m1["2"] = 3;
   cout << m1["2"] << endl;
   return 0;
}
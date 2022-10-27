#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>

using namespace std;
int main(void)
{
   vector<int> v1;
   v1.emplace_back(1);
   v1.emplace_back(1);
   vector<int>::iterator it = v1.end();
   cout << *it << endl;
   deque<int> d1;
   return 0;
}
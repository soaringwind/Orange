#include <iostream>
#include <fstream>


using namespace std;
int main() {
    ifstream inFile;
    inFile.open("./test.txt", ios::in);
    char str1[4];
    // inFile.read(str1, sizeof(str1));
    inFile.getline(str1, sizeof(str1));
    cout << str1 << endl;
}
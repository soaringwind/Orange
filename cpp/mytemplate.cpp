#include <iostream>
#include <string>

template<typename T>
void swap(T *temVal, int len) {
    // 拿到a实际的内存地址
    T tem;
    std::cout << temVal << std::endl;
    for (int i=0; i<len/2; i++) {
        tem = temVal[i];
        temVal[i] = temVal[len-i-1];
        temVal[len-i-1] = tem;
    }
    std::cout << *temVal << std::endl;
}

int main() {
    // long long a = 413241231132441234; 
    // swap(&a, sizeof(a));
    // long long *b = &a;
    // std::cout << &a << std::endl;
    // swap(&a, sizeof(a));
    std::cout << typeid(std::string).name() << std::endl;
    return 0;
}
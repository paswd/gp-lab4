#include <iostream>
#include <ctime>

using namespace std;

int main(void) {
	uint32_t n = 1000;
	uint32_t m = 1000;
	uint32_t k = 1000;

	cout << n << " " << m << " " << k << endl;

	for (uint32_t i = 0; i < n; i++) {
		for (uint32_t j = 0; j < m; j++) {
			cout << rand() % 100 << " ";
		}
		cout << endl;
	}

	for (uint32_t i = 0; i < n; i++) {
		for (uint32_t j = 0; j < k; j++) {
			cout << rand() % 100 << " ";
		}
		cout << endl;
	}
}
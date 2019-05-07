#include <iostream>

using namespace std;

int main(void) {
	float a = 1.;
	int cnt = 0;

	while (a != a / 2.) {
		a /= 10.;

		cnt++;
	}
	cout << cnt << endl;

	return 0;
}
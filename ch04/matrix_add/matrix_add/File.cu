#include <cstdio>

int main() {
	//host-side
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = {0};

	//make a,b matrices
	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < WIDTH; y++) {
			a[x][y] = x * 10 + y;
			b[x][y] = (x * 10 + x) * 100;
			c[x][y] = a[x][y] + b[x][y];
		}
	}

	//print
	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < WIDTH; y++) {
			
			printf("%5d", c[x][y]);
		}
		printf("\n");
	}
	return 0;
}
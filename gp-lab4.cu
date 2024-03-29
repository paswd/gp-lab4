#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
//#include <ctime>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
//#include "../lib/cuPrintf.cu"

using namespace std;

typedef double TNum;

#define CSC(call) do {      \
    cudaError_t e = call;   \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n"\
        , __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(0);            \
    }                       \
} while(0)

//#define EPS .0000001;
//const int32_t BLOCK_DIM = 32;

struct Comparator {
	__host__ __device__ bool operator()(TNum a, TNum b) {
		return a < b; 
	}
};

__constant__ int32_t SIZE_N[1];
__constant__ int32_t SIZE_M[1];
__constant__ int32_t SIZE_K[1];

struct Position {
	int32_t Row;
	int32_t Col;
};

#define IsCorrectPos(i, j, height, width) (i < height && j < width)
#define GetLinearPosition(i, j, height, width) (IsCorrectPos(i, j, height, width) ? \
	(j * height + i) : -1)


__global__ void SwapRows(TNum *a, TNum *b, int32_t row1, int32_t row2, int32_t shift) {
	int32_t begin = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t offset = gridDim.x * blockDim.x;

	int32_t col;
	TNum tmp;
	for (col = begin + shift; col < *SIZE_M; col += offset) {
		tmp = a[GetLinearPosition(row1, col, *SIZE_N, *SIZE_M)];
		a[GetLinearPosition(row1, col, *SIZE_N, *SIZE_M)] = a[GetLinearPosition(row2, col, *SIZE_N, *SIZE_M)];
		a[GetLinearPosition(row2, col, *SIZE_N, *SIZE_M)] = tmp;
	}
	for (col = begin; col < *SIZE_K; col += offset) {
		tmp = b[GetLinearPosition(row1, col, *SIZE_N, *SIZE_K)];
		b[GetLinearPosition(row1, col, *SIZE_N, *SIZE_K)] = b[GetLinearPosition(row2, col, *SIZE_N, *SIZE_K)];
		b[GetLinearPosition(row2, col, *SIZE_N, *SIZE_K)] = tmp;
	}
}

__global__ void Normalize(TNum *a, TNum *b, int32_t row, int32_t shift) {
	if (!(abs(a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)]) > .0000001)) {
		return;
	}
	int32_t begin = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t offset = gridDim.x * blockDim.x;

	int32_t col;
	for (col = begin + shift + 1; col < *SIZE_M; col += offset) {
		a[GetLinearPosition(row, col, *SIZE_N, *SIZE_M)] /=
			a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)];
	}
	for (col = begin; col < *SIZE_K; col += offset) {
		b[GetLinearPosition(row, col, *SIZE_N, *SIZE_K)] /=
			a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)];
	}
}

__global__ void GaussFirst(TNum *a, TNum *b, int32_t row, int32_t shift) {
	if (!(abs(a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)]) > .0000001)) {
		return;
	}
	/*Position begin = SetPosition(blockDim.x * blockIdx.x + threadIdx.x,
								blockDim.y * blockIdx.y + threadIdx.y);
	Position offset = SetPosition(blockDim.x * gridDim.x, blockDim.y * gridDim.y);*/
	//Position curr = begin;

	int32_t beginRow = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t beginCol = blockDim.y * blockIdx.y + threadIdx.y;

	int32_t offsetRow = blockDim.x * gridDim.x;
	int32_t offsetCol = blockDim.y * gridDim.y;
	Position curr;

	//TNum head;
	for (curr.Row = beginRow + row + 1; curr.Row < *SIZE_N; curr.Row += offsetRow) {
		//head = a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		if (!(abs(a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)]) > .0000001)) {
			continue;
		}
		for (curr.Col = beginCol + shift + 1; curr.Col < *SIZE_M; curr.Col += offsetCol) {
			//cuPrintf("\nA\n");
			a[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_M)] -= 
				a[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_M)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		}
		for (curr.Col = beginCol; curr.Col < *SIZE_K; curr.Col += offsetCol) {
			//cuPrintf("\nB\n");
			b[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_K)] -= 
				b[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_K)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		}

		//cuPrintf("\nMAX = %ld\n", max(*SIZE_M, *SIZE_K));

		/*for (curr.Col = beginCol; curr.Col < max(*SIZE_M - shift - 1, *SIZE_K); curr.Col += offsetCol) {
			//cuPrintf("\nSTEP %d\n", curr.Col);
			//cuPrintf("%d >= %d + %d + 1 && %d < %d\n", curr.Col, beginCol, shift, curr.Col, *SIZE_M);
			if (curr.Col < *SIZE_M - shift - 1) {
				//cuPrintf("\nA\n");
				a[GetLinearPosition(curr.Row, (curr.Col + shift + 1), *SIZE_N, *SIZE_M)] -= 
					a[GetLinearPosition(row, (curr.Col + shift + 1), *SIZE_N, *SIZE_M)] *
					a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
			}
			if (curr.Col < *SIZE_K) {
				//cuPrintf("\nB\n");
				b[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_K)] -= 
					b[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_K)] *
					a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
			}
		}*/
	}
}

__global__ void GaussSecond(TNum *a, TNum *b, int32_t row, int32_t shift) {
	/*Position begin = SetPosition(blockDim.x * blockIdx.x + threadIdx.x,
								blockDim.y * blockIdx.y + threadIdx.y);
	Position offset = SetPosition(blockDim.x * gridDim.x, blockDim.y * gridDim.y);*/

	int32_t beginRow = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t beginCol = blockDim.y * blockIdx.y + threadIdx.y;

	int32_t offsetRow = blockDim.x * gridDim.x;
	int32_t offsetCol = blockDim.y * gridDim.y;

	Position curr;

	for (curr.Row = row - 1 - beginRow; curr.Row >= 0; curr.Row -= offsetRow) {
		/*for (curr.Col = begin.Col + shift; curr.Col < *SIZE_M; curr.Col += offset.Col) {
			a[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_M)] -=
				a[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_M)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];	
		}*/
		for (curr.Col = beginCol; curr.Col < *SIZE_K; curr.Col += offsetCol) {
			b[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_K)] -= 
				b[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_K)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		}
	}
}
/*__host__ void GaussSecondCPU(TNum *a, TNum *b, int32_t row, int32_t shift) {
	Position curr;

	for (curr.Row = row - 1; curr.Row >= 0; curr.Row--) {
		for (curr.Col = shift; curr.Col >= 0; curr.Col -= offset.Col) {
			a[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_M)] -=
				a[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_M)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];	
		}
		for (curr.Col = begin.Col; curr.Col >= 0; curr.Col -= offset.Col) {
			b[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_K)] -= 
				b[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_K)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		}
	}
}*/

__host__ void InputMatrix(TNum *matrix, int32_t height, int32_t width) {
	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			//cin >> matrix[GetLinearPosition(i, j, height, width)];
			scanf("%le", matrix + GetLinearPosition(i, j, height, width));
		}
	}
}

__host__ void PrintMatrix(TNum *matrix, int32_t height, int32_t width) {
	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			if (j > 0) {
				//cout << " ";
				printf(" ");
			}
			//cout << scientific << matrix[GetLinearPosition(i, j, height, width)];
			printf("%e", matrix[GetLinearPosition(i, j, height, width)]);
		}
		cout << endl;
	}
}


__host__ int main(void) {
	Comparator cmp;
	int32_t n, m, k;
	//cin >> n >> m >> k;
	//scanf("%d%d%d", &n, &m, &k);
	scanf("%d", &n);
	scanf("%d", &m);
	scanf("%d", &k);
	///cout << n << " " << m << " " << k << endl;

	CSC(cudaMemcpyToSymbol(SIZE_N, &n, sizeof(int32_t)));
	CSC(cudaMemcpyToSymbol(SIZE_M, &m, sizeof(int32_t)));
	CSC(cudaMemcpyToSymbol(SIZE_K, &k, sizeof(int32_t)));

	TNum *a = new TNum[n * m];
	TNum *b = new TNum[n * k];
	//bool *is_success = new bool;

	InputMatrix(a, n, m);
	InputMatrix(b, n, k);


	TNum *cuda_a;
	TNum *cuda_b;

	//bool *cuda_is_success;

	CSC(cudaMalloc((void**) &cuda_a, sizeof(TNum) * n * m));
	CSC(cudaMalloc((void**) &cuda_b, sizeof(TNum) * n * k));
	//CSC(cudaMalloc((void**) &cuda_is_success, sizeof(bool)));

	CSC(cudaMemcpy(cuda_a, a, sizeof(TNum) * n * m, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(cuda_b, b, sizeof(TNum) * n * k, cudaMemcpyHostToDevice));

	int32_t row = 0;
	int32_t *shifts = new int32_t[n];

	//cudaPrintfInit();
	
	memset(shifts, 0, n * sizeof(int32_t));

	/*dim3 threads_per_block(n, m);
	dim3 blocks_per_grid(1, 1);

	if (n * m > BLOCK_DIM * BLOCK_DIM){
		threads_per_block.x = BLOCK_DIM;
		threads_per_block.y = BLOCK_DIM;
		blocks_per_grid.x = ceil((double) (n) / (double)(threads_per_block.x));
		blocks_per_grid.y = ceil((double) (m) / (double)(threads_per_block.y));
	}*/

	for (int32_t col = 0; col < m && row < n; col++) {
		/*CSC(cudaMemcpy(a, cuda_a, sizeof(TNum) * n * m, cudaMemcpyDeviceToHost));
		CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));
		PrintMatrix(a, n, m);
		cout << "---" << endl;
		PrintMatrix(b, n, k);
		cout << "___" << endl;*/
		if (row < n - 1) {
			thrust::device_ptr <TNum> cuda_a_begin = thrust::device_pointer_cast(cuda_a);
			thrust::device_ptr <TNum> cuda_a_max = thrust::max_element(
				cuda_a_begin + GetLinearPosition(row, col, n, m),
				cuda_a_begin + (col + 1) * n, cmp);
			int32_t row_max_pos = cuda_a_max - cuda_a_begin - GetLinearPosition(0, col, n, m);

			//TNum row_value, max_value;
			//cout << sizeof(TNum) << endl;
			//cout << cuda_a << " : " << cuda_a + n * m * sizeof(TNum) << endl;
			//cout <<  cuda_a + sizeof(TNum) * GetLinearPosition(row, col, n, m) << " : " <<
				//cuda_a + sizeof(TNum) * GetLinearPosition(row_max_pos, col, n, m) << endl;

			/*CSC(cudaMemcpy(&row_value, cuda_a + GetLinearPosition(row, col, n, m),
				sizeof(TNum), cudaMemcpyDeviceToHost));
			CSC(cudaMemcpy(&max_value, cuda_a + GetLinearPosition(row_max_pos, col, n, m),
				sizeof(TNum), cudaMemcpyDeviceToHost));

			TNum curr = row_value;*/
			//cout << curr << " : " << max_value << endl;

			if (row_max_pos != row) {
				SwapRows<<<dim3(1024), dim3(1024)>>>(cuda_a, cuda_b, row, row_max_pos, col);
				//curr = max_value;
			}
			/*if (!(abs(curr) > .0000001)) {
				//cout << "CURR = " << curr << endl;
				//cout << "OUT1" << endl;
				continue;
			}*/
		}/* else {
			TNum curr;
			//cout << GetLinearPosition(row, col, n, m) << endl;
			//cout << row << ":" << col << endl;
			CSC(cudaMemcpy(&curr, cuda_a + GetLinearPosition(row, col, n, m),
				sizeof(TNum), cudaMemcpyDeviceToHost));
			if (!(abs(curr) > .0000001)) {
				//cout << "OUT2" << endl;
				continue;
			}
		}*/
		/*CSC(cudaMemcpy(a, cuda_a, sizeof(TNum) * n * m, cudaMemcpyDeviceToHost));
		CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));
		cout << "Col: " << col << endl;
		PrintMatrix(a, n, m);
		cout << "---" << endl;
		PrintMatrix(b, n, k);
		cout << "~~~" << endl;*/

		//cudaPrintfInit();
		Normalize<<<dim3(1024), dim3(1024)>>>(cuda_a, cuda_b, row, col);
		//bool is_success;
		TNum curr;
		CSC(cudaMemcpy(&curr, cuda_a + GetLinearPosition(row, col, n, m),
			sizeof(TNum), cudaMemcpyDeviceToHost));
		if (!(abs(curr) > .0000001)) {
			//cout << "OUT2" << endl;
			continue;
		}
		//cout << (*is_success ? "true" : "false") << endl;
		//cudaPrintfDisplay(stdout, true);
    	//cudaPrintfEnd();

		/*CSC(cudaMemcpy(a, cuda_a, sizeof(TNum) * n * m, cudaMemcpyDeviceToHost));
		CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));
		PrintMatrix(a, n, m);
		cout << "---" << endl;
		PrintMatrix(b, n, k);
		cout << "+++" << endl;*/


		if (row < n - 1) {
			GaussFirst<<<dim3(32, 32), dim3(32, 32)>>>(cuda_a, cuda_b, row, col);
		}
		//cout << shifts[row] << " -> " << col << endl;
		shifts[row] = col;
		row++;

		/*CSC(cudaMemcpy(a, cuda_a, sizeof(TNum) * n * m, cudaMemcpyDeviceToHost));
		CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));
		PrintMatrix(a, n, m);
		cout << "---" << endl;
		PrintMatrix(b, n, k);
		cout << "===" << endl << endl;*/
	}
	/*cout << "NEXT!!" << endl;
	CSC(cudaMemcpy(a, cuda_a, sizeof(TNum) * n * m, cudaMemcpyDeviceToHost));
	CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));
	PrintMatrix(a, n, m);
	cout << "---" << endl;
	PrintMatrix(b, n, k);
	cout << "===" << endl << endl;*/

	for (int32_t row_curr = row - 1; row_curr >= 0; row_curr--) {
		if (row_curr > 0) {
			GaussSecond<<<dim3(32, 32), dim3(32, 32)>>>(cuda_a, cuda_b, row_curr, shifts[row_curr]);
		}
		/*CSC(cudaMemcpy(a, cuda_a, sizeof(TNum) * n * m, cudaMemcpyDeviceToHost));
		CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));
		PrintMatrix(a, n, m);
		cout << "---" << endl;
		PrintMatrix(b, n, k);
		cout << "===" << endl << endl;*/
	}

	//int32_t *cuda_shifts;
	//cudaMalloc((void**) &cuda_shifts, sizeof(int32_t) * row);
	//cudaMemcpy(cuda_shifts, shifts, sizeof(int32_t) * row, cudaMemcpyHostToDevice);


	//GetResult<<<dim3(32, 32), dim3(32, 32)>>>(cuda_b, cuda_x, cuda_shifts, row, );

	//cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();

	/*cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);
	cudaEventDestroy(syncEvent);*/

	//Calculating end

	CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));

	CSC(cudaFree(cuda_a));
	CSC(cudaFree(cuda_b));
	//cudaFree(cuda_x);

	//PrintMatrix(cuda_b, shifts, m, k);

	TNum zero = 0.;

	int32_t untill = 0;
	if (row > 0) {
		untill = shifts[0];
	}
	int32_t rows_cnt = 0;
	for (int32_t i = 0; i < untill; i++) {
		for (int32_t j = 0; j < k; j++) {
			//cout << "1: " << shifts[0] << "::" << i << ":" << j << endl;
			if (j > 0) {
				//cout << " ";
				printf(" ");
			}
			//cout << scientific << zero;
			printf("%e", zero);
		}
		rows_cnt++;
		//cout << endl;
		printf("\n");
	}

	//cout << row << endl;

	for (int32_t i = 0; i < row; i++) {
		if (i > 0) {
			for (int32_t ii = 0; ii < shifts[i] - shifts[i - 1] - 1; ii++) {
				for (int32_t j = 0; j < k; j++) {
					if (j > 0) {
						//cout << " ";
						printf(" ");
					}
					//cout << "2: " << i << ":" << j << endl;
					//cout << scientific << zero;
					printf("%e", zero);
				}
				rows_cnt++;
				//cout << endl;
				printf("\n");
			}
		}
		for (int32_t j = 0; j < k; j++) {
			if (j > 0) {
				//cout << " ";
				printf(" ");
			}
			//cout << "3: " << i << ":" << j << endl;
			//cout << scientific << b[GetLinearPosition(i, j, n, k)];
			printf("%e", b[GetLinearPosition(i, j, n, k)]);
		}
		rows_cnt++;
		//cout << endl;
		printf("\n");
	}

	//cout << "TEST0" << endl;
	//cout << shifts[0] << endl;

	//untill = m - shifts[max(0, (int32_t) row - 1)];

	for (int32_t i = 0; i < m - rows_cnt; i++) {
		for (int32_t j = 0; j < k; j++) {
			if (j > 0) {
				//cout << " ";
				printf(" ");
			}
			//cout << "4: " << i << ":" << j << endl;
			//cout << scientific << zero;
			printf("%e", zero);
		}
		//cout << endl;
		printf("\n");
	}
	//cout << "TEST1" << endl;
	/*cout << "SHIFTS:\n";
	for (int32_t i = 0; i < row; i++) {
		cout << shifts[i] << endl;
	}*/

	delete [] shifts;

	delete [] a;
	delete [] b;
	//delete [] cuda_x;

	return 0;
}
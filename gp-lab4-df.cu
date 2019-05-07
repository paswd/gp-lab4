#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
//#include <ctime>
#include <vector>
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

struct Comparator {
	__host__ __device__ bool operator()(TNum a, TNum b) {
		return a < b; 
	}
};

__constant__ uint32_t SIZE_N[1];
__constant__ uint32_t SIZE_M[1];
__constant__ uint32_t SIZE_K[1];

struct Position {
	int32_t Row;
	int32_t Col;
};

__device__ __host__ Position SetPosition(int32_t i, int32_t j) {
	Position pos;
	pos.Row = i;
	pos.Col = j;
	return pos;
}
__device__ __host__ Position SetPosition(uint32_t i, uint32_t j) {
	return SetPosition((int32_t) i, (int32_t) j);
}


__device__ __host__ bool IsCorrectPos(Position pos, uint32_t height, uint32_t width) {
	return (pos.Row >= 0 && pos.Col >= 0 && pos.Row < (int32_t) height && pos.Col < (int32_t) width);
}
__device__ __host__ int32_t GetLinearPosition(Position pos, uint32_t height, uint32_t width) {
	return (IsCorrectPos(pos, height, width)) ? (pos.Col * (int32_t) height + pos.Row) : -1;
}

__device__ __host__ bool IsCorrectPos(uint32_t i, uint32_t j, uint32_t height, uint32_t width) {
	return (i < height && j < width);
}
__device__ __host__ int32_t GetLinearPosition(uint32_t i, uint32_t j, uint32_t height, uint32_t width) {
	return (IsCorrectPos(i, j, height, width) ? ((int32_t) j * (int32_t) height + (int32_t) i) : -1);
}

/*__global__ void SwapVector(TNum *a, TNum *b, uint32_t length, uint32_t shift) {
	uint32_t begin = blockDim.x * blockIdx.x + threadIdx.x + shift;
	uint32_t offset = gridDim.x * blockDim.x;

	TNum tmp;

	for (uint32_t i = begin, i < length, i += offset) {

	}
}*/

__global__ void SwapRows(TNum *a, TNum *b, uint32_t row1, uint32_t row2, uint32_t shift) {
	//uint32_t begin = blockDim.x * blockIdx.x + threadIdx.x;
	//uint32_t offset = gridDim.x * blockDim.x;

	uint32_t col;
	TNum tmp;
	for (col = (blockDim.x * blockIdx.x + threadIdx.x) + shift; col < *SIZE_M; col += gridDim.x * blockDim.x) {
		tmp = a[GetLinearPosition(row1, col, *SIZE_N, *SIZE_M)];
		a[GetLinearPosition(row1, col, *SIZE_N, *SIZE_M)] = a[GetLinearPosition(row2, col, *SIZE_N, *SIZE_M)];
		a[GetLinearPosition(row2, col, *SIZE_N, *SIZE_M)] = tmp;
	}
	for (col = blockDim.x * blockIdx.x + threadIdx.x; col < *SIZE_K; col += gridDim.x * blockDim.x) {
		tmp = b[GetLinearPosition(row1, col, *SIZE_N, *SIZE_K)];
		b[GetLinearPosition(row1, col, *SIZE_N, *SIZE_K)] = b[GetLinearPosition(row2, col, *SIZE_N, *SIZE_K)];
		b[GetLinearPosition(row2, col, *SIZE_N, *SIZE_K)] = tmp;
	}
}

__global__ void Normalize(TNum *a, TNum *b, uint32_t row, uint32_t shift) {
	if (!(abs(a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)]) > .0000001)) {
		return;
	}
	//uint32_t begin = blockDim.x * blockIdx.x + threadIdx.x;
	//uint32_t offset = gridDim.x * blockDim.x;

	uint32_t col;
	for (col = (blockDim.x * blockIdx.x + threadIdx.x) + shift + 1; col < *SIZE_M; col += gridDim.x * blockDim.x) {
		a[GetLinearPosition(row, col, *SIZE_N, *SIZE_M)] /=
			a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)];
	}
	for (col = blockDim.x * blockIdx.x + threadIdx.x; col < *SIZE_K; col += gridDim.x * blockDim.x) {
		b[GetLinearPosition(row, col, *SIZE_N, *SIZE_K)] /=
			a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)];
	}
}

__global__ void GaussFirst(TNum *a, TNum *b, uint32_t row, uint32_t shift) {
	if (!(abs(a[GetLinearPosition(row, shift, *SIZE_N, *SIZE_M)]) > .0000001)) {
		return;
	}
	/*Position begin = SetPosition(blockDim.x * blockIdx.x + threadIdx.x,
								blockDim.y * blockIdx.y + threadIdx.y);
	Position offset = SetPosition(blockDim.x * gridDim.x, blockDim.y * gridDim.y);*/
	//Position curr = begin;

	/*int32_t beginRow = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t beginCol = blockDim.y * blockIdx.y + threadIdx.y;

	int32_t offsetRow = blockDim.x * gridDim.x;
	int32_t offsetCol = blockDim.y * gridDim.y;*/
	Position curr;

	//TNum head;
	for (curr.Row = (blockDim.x * blockIdx.x + threadIdx.x) + row + 1; curr.Row < *SIZE_N; curr.Row += blockDim.x * gridDim.x) {
		//head = a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		if (!(abs(a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)]) > .0000001)) {
			continue;
		}
		for (curr.Col = (blockDim.y * blockIdx.y + threadIdx.y) + shift + 1; curr.Col < *SIZE_M; curr.Col += blockDim.y * gridDim.y) {
			a[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_M)] -= 
				a[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_M)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		}
		for (curr.Col = blockDim.y * blockIdx.y + threadIdx.y; curr.Col < *SIZE_K; curr.Col += blockDim.y * gridDim.y) {
			b[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_K)] -= 
				b[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_K)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		}
	}
}

__global__ void GaussSecond(TNum *a, TNum *b, uint32_t row, uint32_t shift) {
	/*Position begin = SetPosition(blockDim.x * blockIdx.x + threadIdx.x,
								blockDim.y * blockIdx.y + threadIdx.y);
	Position offset = SetPosition(blockDim.x * gridDim.x, blockDim.y * gridDim.y);*/

	/*int32_t beginRow = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t beginCol = blockDim.y * blockIdx.y + threadIdx.y;

	int32_t offsetRow = blockDim.x * gridDim.x;
	int32_t offsetCol = blockDim.y * gridDim.y;*/

	Position curr;

	for (curr.Row = row - 1 - (blockDim.x * blockIdx.x + threadIdx.x); curr.Row >= 0; curr.Row -= blockDim.x * gridDim.x) {
		/*for (curr.Col = begin.Col + shift; curr.Col < *SIZE_M; curr.Col += offset.Col) {
			a[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_M)] -=
				a[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_M)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];	
		}*/
		for (curr.Col = blockDim.y * blockIdx.y + threadIdx.y; curr.Col < *SIZE_K; curr.Col += blockDim.y * gridDim.y) {
			b[GetLinearPosition(curr.Row, curr.Col, *SIZE_N, *SIZE_K)] -= 
				b[GetLinearPosition(row, curr.Col, *SIZE_N, *SIZE_K)] *
				a[GetLinearPosition(curr.Row, shift, *SIZE_N, *SIZE_M)];
		}
	}
}
/*__host__ void GaussSecondCPU(TNum *a, TNum *b, uint32_t row, uint32_t shift) {
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

/*__global__ void GetResult(TNum *b, TNum *x, uint32_t *shifts, uint32_t rows) {
	Position begin = SetPosition(blockDim.x * blockIdx.x + threadIdx.x,
								blockDim.y * blockIdx.y + threadIdx.y);
	Position offset = SetPosition(blockDim.x * gridDim.x, blockDim.y * gridDim.y);
	Position curr;

	for (curr.Row = begin.Row; curr.Row < rows; curr.Rows++) {
		for (curr.Col = begin.Col; curr.Col < *SIZE_K; curr.Col++) {

		}
	}

}*/

__host__ void InputMatrix(TNum *matrix, uint32_t height, uint32_t width) {
	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			cin >> matrix[GetLinearPosition(i, j, height, width)];
		}
	}
}

__host__ void PrintMatrix(TNum *matrix, uint32_t height, uint32_t width) {
	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			if (j > 0) {
				cout << " ";
			}
			cout << scientific << matrix[GetLinearPosition(SetPosition(i, j), height, width)];
		}
		cout << endl;
	}
}


__host__ int main(void) {
	Comparator cmp;
	uint32_t n, m, k;
	cin >> n >> m >> k;

	CSC(cudaMemcpyToSymbol(SIZE_N, &n, sizeof(uint32_t)));
	CSC(cudaMemcpyToSymbol(SIZE_M, &m, sizeof(uint32_t)));
	CSC(cudaMemcpyToSymbol(SIZE_K, &k, sizeof(uint32_t)));

	TNum *a = new TNum[n * m];
	TNum *b = new TNum[n * k];

	InputMatrix(a, n, m);
	InputMatrix(b, n, k);


	TNum *cuda_a;
	TNum *cuda_b;

	CSC(cudaMalloc((void**) &cuda_a, sizeof(TNum) * n * m));
	CSC(cudaMalloc((void**) &cuda_b, sizeof(TNum) * n * k));

	CSC(cudaMemcpy(cuda_a, a, sizeof(TNum) * n * m, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(cuda_b, b, sizeof(TNum) * n * k, cudaMemcpyHostToDevice));

	uint32_t row = 0;
	uint32_t *shifts = new uint32_t[n];
	memset(shifts, 0, n * sizeof(uint32_t));

	for (uint32_t col = 0; col < m && row < n; col++) {
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
			uint32_t row_max_pos = cuda_a_max - cuda_a_begin - GetLinearPosition(0, col, n, m);

			TNum row_value, max_value;
			//cout << sizeof(TNum) << endl;
			//cout << cuda_a << " : " << cuda_a + n * m * sizeof(TNum) << endl;
			//cout <<  cuda_a + sizeof(TNum) * GetLinearPosition(row, col, n, m) << " : " <<
				//cuda_a + sizeof(TNum) * GetLinearPosition(row_max_pos, col, n, m) << endl;

			CSC(cudaMemcpy(&row_value, cuda_a + GetLinearPosition(row, col, n, m),
				sizeof(TNum), cudaMemcpyDeviceToHost));
			CSC(cudaMemcpy(&max_value, cuda_a + GetLinearPosition(row_max_pos, col, n, m),
				sizeof(TNum), cudaMemcpyDeviceToHost));

			TNum curr = row_value;
			//cout << curr << " : " << max_value << endl;

			if (row_max_pos != row && row_value < max_value) {
				SwapRows<<<dim3(1024), dim3(1024)>>>(cuda_a, cuda_b, row, row_max_pos, col);
				curr = max_value;
			}
			if (!(abs(curr) > .0000001)) {
				//cout << "CURR = " << curr << endl;
				//cout << "OUT1" << endl;
				continue;
			}
		} else {
			TNum curr;
			//cout << GetLinearPosition(row, col, n, m) << endl;
			//cout << row << ":" << col << endl;
			CSC(cudaMemcpy(&curr, cuda_a + GetLinearPosition(row, col, n, m),
				sizeof(TNum), cudaMemcpyDeviceToHost));
			if (!(abs(curr) > .0000001)) {
				//cout << "OUT2" << endl;
				continue;
			}
		}
		/*CSC(cudaMemcpy(a, cuda_a, sizeof(TNum) * n * m, cudaMemcpyDeviceToHost));
		CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));
		cout << "Col: " << col << endl;
		PrintMatrix(a, n, m);
		cout << "---" << endl;
		PrintMatrix(b, n, k);
		cout << "~~~" << endl;*/

		//cudaPrintfInit();
		Normalize<<<dim3(1024), dim3(1024)>>>(cuda_a, cuda_b, row, col);
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

	for (uint32_t i = row; i > 0; i--) {
		uint32_t row_curr = i - 1;
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

	//uint32_t *cuda_shifts;
	//cudaMalloc((void**) &cuda_shifts, sizeof(uint32_t) * row);
	//cudaMemcpy(cuda_shifts, shifts, sizeof(uint32_t) * row, cudaMemcpyHostToDevice);


	//GetResult<<<dim3(32, 32), dim3(32, 32)>>>(cuda_b, cuda_x, cuda_shifts, row, );

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);
	cudaEventDestroy(syncEvent);

	//Calculating end

	CSC(cudaMemcpy(b, cuda_b, sizeof(TNum) * n * k, cudaMemcpyDeviceToHost));

	CSC(cudaFree(cuda_a));
	CSC(cudaFree(cuda_b));
	//cudaFree(cuda_x);

	//PrintMatrix(cuda_b, shifts, m, k);

	TNum zero = 0.;

	uint32_t untill = 0;
	if (row > 0) {
		untill = shifts[0];
	}
	uint32_t rows_cnt = 0;
	for (uint32_t i = 0; i < untill; i++) {
		for (uint32_t j = 0; j < k; j++) {
			//cout << "1: " << shifts[0] << "::" << i << ":" << j << endl;
			if (j > 0) {
				cout << " ";
			}
			cout << scientific << zero;
		}
		rows_cnt++;
		cout << endl;
	}

	//cout << row << endl;

	for (uint32_t i = 0; i < row; i++) {
		if (i > 0) {
			for (uint32_t ii = 0; ii < shifts[i] - shifts[i - 1] - 1; ii++) {
				for (uint32_t j = 0; j < k; j++) {
					if (j > 0) {
						cout << " ";
					}
					//cout << "2: " << i << ":" << j << endl;
					cout << scientific << zero;
				}
				rows_cnt++;
				cout << endl;
			}
		}
		for (uint32_t j = 0; j < k; j++) {
			if (j > 0) {
				cout << " ";
			}
			//cout << "3: " << i << ":" << j << endl;
			cout << scientific << b[GetLinearPosition(i, j, n, k)];
		}
		rows_cnt++;
		cout << endl;
	}

	//cout << "TEST0" << endl;
	//cout << shifts[0] << endl;

	//untill = m - shifts[max(0, (int32_t) row - 1)];

	for (uint32_t i = 0; i < m - rows_cnt; i++) {
		for (uint32_t j = 0; j < k; j++) {
			if (j > 0) {
				cout << " ";
			}
			//cout << "4: " << i << ":" << j << endl;
			cout << scientific << zero;
		}
		cout << endl;
	}
	//cout << "TEST1" << endl;
	/*cout << "SHIFTS:\n";
	for (uint32_t i = 0; i < row; i++) {
		cout << shifts[i] << endl;
	}*/

	delete [] shifts;

	delete [] a;
	delete [] b;	
	//delete [] cuda_x;

	return 0;
}
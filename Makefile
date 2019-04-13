FLAGS=-ccbin clang++-3.8 -std=c++11 --compiler-options -stdlib=libc++ -Wno-deprecated-gpu-targets -lm
COMPILLER=nvcc

#all: lib start
all: start

#start: main.o
#	$(COMPILLER) $(FLAGS) -o da-lab4 main.o -L. lib/lib-z-search.a

start: gp-lab4.cu
	$(COMPILLER) $(FLAGS) -o gp-lab4 gp-lab4.cu

bench: bench.cu
	$(COMPILLER) $(FLAGS) -o bench bench.cu

clean:
	rm gp-lab4

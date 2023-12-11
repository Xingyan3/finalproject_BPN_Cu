NVCC        = nvcc
NVCC_FLAGS  = -O3 
LD_FLAGS    = -lcudart 
EXE	        = bpn
OBJ	        = bpn.o

default: $(EXE)

bpn.o: bpn.cu
	$(NVCC) -c -o $@ bpn.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

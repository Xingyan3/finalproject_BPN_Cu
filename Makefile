NVCC        = nvcc
NVCC_FLAGS  = -O3 
LD_FLAGS    = -lcudart 
EXE	        = bpn
OBJ	        = bpn.o support.o

default: $(EXE)

main.o: bpn.cu support.h
	$(NVCC) -c -o $@ bpn.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

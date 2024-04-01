GPU=1
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=0

#ARCH= -gencode arch=compute_30,code=sm_30 \
#      -gencode arch=compute_35,code=sm_35 \
#      -gencode arch=compute_50,code=[sm_50,compute_50] \
#      -gencode arch=compute_52,code=[sm_52,compute_52]
ARCH= -gencode arch=compute_80,code=sm_80  		\
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_70,code=[sm_70,compute_70]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

# SYCL: Use intel clang and clang++ to compile.
CC=clang
CPP=clang++
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
#LDFLAGS= -lm -pthread -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lOpenCL -lpthread -ldl
LDFLAGS= -lm -pthread -lonemkl
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wpointer-arith -fPIC 
SYCLFLAGS=-fsycl -std=c++17 

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
# COMMON+= -DGPU -I/usr/local/cuda/include/
COMMON+= -DGPU
CFLAGS+= -DGPU 
# LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
SYCLFLAGS+=-fsycl-targets=nvptx64-nvidia-cuda 
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN -I/export/users/zhaojiam/pkg/cudnn-linux-x86_64-9.0.0.312_cuda12-archive/include
CFLAGS+= -DCUDNN -DCUDNN_MAJOR=9
LDFLAGS+= -lcudnn -L/export/users/zhaojiam/pkg/cudnn-linux-x86_64-9.0.0.312_cuda12-archive/lib

COMMON+= -I/nfs/shm/proj/icl/cmplrarch/comp/onednnl/lin/2024.1.0.561/dnnl/latest/include
LDFLAGS+= -ldnnl -L/nfs/shm/proj/icl/cmplrarch/comp/onednnl/lin/2024.1.0.561/dnnl/latest/lib
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o debug.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

# $(warning "EXECOBJ" $(EXECOBJ))
# $(warning "OBJS" $(OBJS))
# $(warning "DEPS" $(DEPS))

all: obj $(EXEC)
# all: obj backup results $(SLIB) $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CPP) $(COMMON) $(CFLAGS) $(SYCLFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB) 

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

# $(SLIB): $(OBJS)
# 	$(CC) $(CFLAGS)-shared $^ -o $@ $(LDFLAGS)

# $(OBJDIR)%.o: %.cpp $(DEPS)
# 	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@
image_opencv.o: ./src/image_opencv.cpp
	$(CPP) $(COMMON) $(CFLAGS) $(SYCLFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.dp.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) $(SYCLFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

# $(OBJDIR)%.o: %.dp.cpp $(DEPS)
# 	$(CPP) $(ARCH) $(COMMON) "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*


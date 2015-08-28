
ROOT_DIR = ../
# Include the definition of tool chain
include $(ROOT_DIR)/tool_chain.def

#---------------------------------------------------------------------------
# Variable definition:
#   1. Cross-compling tools
#   2. Linker script file
#   3. C complier options
#   4. Assembly compiler options
#   5. Linker options
#---------------------------------------------------------------------------
ECHO    = echo
LIB_PATH     = $(ROOT_DIR)/release/lib
UNAME=Linux

GOAL    = svmTrain

CFLAGS += -fopenmp -std=c++0x

CFLAGS += -I./
CFLAGS += -I../
CFLAGS += -I$(ROOT_DIR)/release/include/

ifeq ($(UNAME), Linux)

#LIBS += /usr/lib64/libhdfs.so

LIBS += $(LIB_PATH)/libopencv_calib3d.a
LIBS += $(LIB_PATH)/libopencv_features2d.a
LIBS += $(LIB_PATH)/libopencv_flann.a
LIBS += $(LIB_PATH)/libopencv_imgcodecs.a
LIBS += $(LIB_PATH)/libopencv_objdetect.a
LIBS += $(LIB_PATH)/libopencv_imgproc.a
LIBS += $(LIB_PATH)/libopencv_ml.a
LIBS += $(LIB_PATH)/libopencv_core.a
LIBS += $(LIB_PATH)/libopencv_highgui.a
#LIBS += $(LIB_PATH)/libopencv_ml.a
LIBS += $(LIB_PATH)/libopencv_hal.a
LIBS += $(LIB_PATH)/libopencv_videoio.a

LIBS += $(LIB_PATH)/libIlmImf.a
LIBS += $(LIB_PATH)/liblibwebp.a
LIBS += /usr/lib64/libpng12.so
LIBS += /usr/lib64/libjpeg.a
LIBS += $(LIB_PATH)/liblibtiff.a
LIBS += $(LIB_PATH)/liblibjasper.a
LIBS += $(LIB_PATH)/libippicv.a

#LIBS += /usr/java/jdk1.6.0_31/jre/lib/amd64/server/libjvm.so

#LIBS += $(LIB_PATH)/libzlib.a

else

LIBS += $(LIB_PATH)/libopencv_core.dylib
LIBS += $(LIB_PATH)/libopencv_highgui.dylib
LIBS += $(LIB_PATH)/libopencv_imgproc.dylib
LIBS += $(LIB_PATH)/libopencv_ml.dylib
LIBS += $(LIB_PATH)/libopencv_objdetect.dylib

endif

LFLAGS+= -fopenmp
LFLAGS+= -lstdc++
LFLAGS+= -ldl

#---------------------------------------------------------------------------
# Intermediate objects
#---------------------------------------------------------------------------
#OBJS += svmTraining.o
OBJS += svmFaultTrain.o

#OBJS += svmlight/svm_hideo.o
#---------------------------------------------------------------------------
# Implicit rules
#---------------------------------------------------------------------------
.cpp.o:
	$(MSG_SPLIT_LINE)
	$(MSG_COMPILING)
	@$(CC) $(CFLAGS) -MM -MT $@ -o $*.d $<
	@$(CC) -c $(CFLAGS) -o $*.o $<

.c.o:
	$(MSG_SPLIT_LINE)
	$(MSG_COMPILING)
	@$(CC) $(CFLAGS) -MM -MT $@ -o $*.d $<
	@$(CC) -c $(CFLAGS) -o $*.o $<

.S.o:
	$(MSG_SPLIT_LINE)
	$(MSG_COMPILING)
	@$(CC) $(CFLAGS) -MM -MT $@ -o $*.d $<
	@$(CC) -c $(ASFLAGS) $(CFLAGS) -o $*.o $<

#---------------------------------------------------------------------------
# Explicit ruls
#---------------------------------------------------------------------------
all: $(GOAL)


$(GOAL): $(OBJS)
	$(MSG_SPLIT_LINE)
	@$(CC) -o $(GOAL) $(LFLAGS) $(OBJS) $(LIBS)
	$(MSG_GOAL_OK)

.PHONY: clean
clean:
	$(MSG_SPLIT_LINE)
	rm -f $(OBJS:.o=.d) $(OBJS) $(GOAL)

.PHONY: release
release:all


-include $(OBJS:.o=.d)

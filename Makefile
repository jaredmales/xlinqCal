# xlinqCal build. Requires mxlib (with MILK support), ImageStreamIO, and cfitsio, found via pkg-config mxlib.

CXX ?= g++

GIT_VERSION := $(shell git describe --always --dirty 2>/dev/null || echo unknown)

OPTIMIZE ?= -O3
ifeq ($(DEBUG),1)
  OPTIMIZE = -O0 -g
endif

CXXFLAGS += $(shell pkg-config --cflags mxlib) -std=c++20 -Wall -Wextra $(OPTIMIZE)
CXXFLAGS += -DXLINQCAL_GIT_VERSION=\"$(GIT_VERSION)\" -Isrc -Isim
LDLIBS   += $(shell pkg-config --libs mxlib) -pthread

HEADERS = src/xlinqCal.hpp src/isioStream.hpp src/wfsChannel.hpp src/indiClient.hpp sim/xlinqSim.hpp

all: xlinqCal xlinqSim

xlinqCal: src/xlinqCal.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

xlinqSim: sim/xlinqSim.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

tests/xlinqCal_test: tests/xlinqCal_test.cpp tests/testMain.o $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< tests/testMain.o $(LDLIBS)

tests/testMain.o: tests/testMain.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: tests/xlinqCal_test
	./tests/xlinqCal_test

format:
	clang-format -i src/*.hpp src/*.cpp sim/*.hpp sim/*.cpp tests/xlinqCal_test.cpp

clean:
	rm -f xlinqCal xlinqSim tests/xlinqCal_test tests/*.o

.PHONY: all test format clean

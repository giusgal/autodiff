CXX = g++
CXXFLAGS = -g -I/usr/include/eigen3
LDFLAGS = -lcgraph -lgvc

TARGET = test
SRCS = test.cpp
HEADERS = autodiff.hpp utils.hpp neural.hpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) output_graph.png


CC=g++
CFLAGS=--std=c++11 -I. -Wall -O3

KERAS=keras_model.o

%.o: %.cc
ifneq ($(static_analysis),false)
	cppcheck --error-exitcode=1 $<
	clang-tidy $< -checks=clang-analyzer-*,readability-*,performance-* -- $(CFLAGS)
endif
	$(CC) $(CFLAGS) -o $@ -c $<

all: $(KERAS)

clean:
	rm -rf *.o
	rm -rf *.d
	rm -rf $(KERAS)

-include $(KERAS:%.o=%.d)

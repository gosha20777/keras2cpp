CC=g++
CFLAGS=--std=c++11 -I. -Wall -Werror -MMD -O3 -mtune=core2

KERAS=keras_model.o
TESTS=keras_model_unit_test

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%_test: %_test.o $(KERAS)
	$(CC) -o $@ $< $(KERAS)
	#./$@

all: $(KERAS) $(TESTS)

clean:
	rm -rf *.o
	rm -rf *.d
	rm -rf $(KERAS)
	rm -rf $(TESTS)
	rm -rf test_*

-include $(TESTS:%_test=%_test.d)
-include $(KERAS:%.o=%.d)

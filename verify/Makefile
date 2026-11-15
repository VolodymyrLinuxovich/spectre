CXX = clang++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

verify: verify.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f verify

.PHONY: clean

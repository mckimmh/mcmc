include ../src/Makefile_variables
VPATH = ../include ../src

################################################################################

rwm_bvg.out : $(RWM) mvg.o rwm_bvg.o
	$(CXX) -o $@ $^ $(LIBS)

################################################################################

include ../src/Makefile_obj

rwm_bvg.o : rwm_bvg.cpp log_post.h mcmc.h mvg.h print.h rwm.h
	$(CXX) $(CXXFLAGS) -c rwm_bvg.cpp

.PHONY : clean
clean :
	rm *.out *.o *.txt
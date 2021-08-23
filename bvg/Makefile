include ../src/Makefile_variables
VPATH = ../include ../src

################################################################################

hmc_bvg.out : $(HMC) mvg.o hmc_bvg.o
	$(CXX) $(LIBS) -o $@ $^

################################################################################

include ../src/Makefile_obj

hmc_bvg.o : hmc_bvg.cpp log_post.h mcmc.h hmc.h mvg.h
	$(CXX) $(CXXFLAGS) -c hmc_bvg.cpp
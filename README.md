# Quartz
 A simple semi-classical dynamics library 
 (although it also enables simple Discrete Variable Representation Calculation) 
 that assists my graduation thesis in Zhejiang University. 
 It is also for the purpose of software-engineering practice.
 
 This header-only library uses C++17 standard. 
 'Unfortunately', this library depends on several other external libraries, including:
 
 Armadillo (8.500.0 or later, Required externally)
 
 catchorg/Catch2
 
 Taywee/args
 
 fmtlib/fmt
 
 among which Armadillo is not integrated in the library.
 (Special thanks to these libraries!) 
 Boost is also utilised for 
 json input/output for accompanying exectuables, through which you can quickly work on
 small systems. Nevertheless, this repository is still designed as a header-only
 library. And... well... currently there is no document provided.
 
 It also uses OpenMP for auto-parallelisation,
 while Armadillo already enables parallelisation for linear algebra calculations
 if it is linked to MKL or OpenBLAS.
 
 CMake is utilized for compilation, and you may try

```
$ mkdir build
$ cmake .. -DCMAKE_PREFIX_PATH=/the/path/to/your/Armadillo/installation
$ make -j 4
$ make install
```
which will have this library installed to `/usr/local` directory.
 
 It is designed fully capable with arbitrary dimensions, 
 with highly flexible modules such as several elementary functions 
 enabling analytical calcuation, 
 Runge-Kutta methods, automatic propagations and printing sections and a lot more, 
 aiming to minimize the effort for implementing a new method 
 or enhancing the already implemented ones. 
 The header-only structure with heavy loads of templates and classes 
 is expected to greatly ease the utilization of this library, 
 while loads of tests should guarantee the normal execution of the functions.

`test/propagate_test.cpp` contains basic usage of this library, by specifying 
the initial conditions and potential, then stating the method and corresponding
operator, the way operator is going to be wrapped (e.g. Runge-Kutta-2/4), and 
the printer which specifies how the status is printed on the screen. A typical 
result, using DVR method for a harmonic potential, is as follows:
```
Library: Quartz
version: 0.0.1
============================================================
|Step|            Time |      Positional |        Momentum |
============================================================
     0         0.0000000        0.99999999    -1.1928157e-16
     1              0.01        0.99994999     -0.0099998333
     2              0.02            0.9998      -0.019998667
     3              0.03        0.99955003        -0.0299955
     4              0.04         0.9992001      -0.039989334
     5              0.05        0.99875025      -0.049979169
     6              0.06        0.99820053      -0.059964006
     7              0.07        0.99755099      -0.069942847
     8              0.08         0.9968017      -0.079914694
     9              0.09        0.99595273      -0.089878549
    10               0.1        0.99500416      -0.099833416
============================================================

Quartz terminated normally.

```

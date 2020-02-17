# Quartz
 A simple semi-classical dynamics library 
 (although it also enables simple Discrete Variable Representation Calculation) 
 that assists my graduation thesis in Zhejiang University. 
 It is also for the purpose of software-engineering practice.
 
 This header-only library uses C++17 standard. 
 There are currently two third-party libraries used, Armadillo and Catch2, 
 among which Armadillo is not integrated in the library. 
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
 with highly flexible modules such as several elementary functions enabling analytical calcuation, 
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

            Time |      Positional |        Momentum |
======================================================
                 0        0.99999998                 0
              0.01        0.99994999     -0.0099991688
              0.02        0.99980001      -0.019997312
              0.03        0.99955008      -0.029993447
              0.04         0.9992002      -0.039986586
              0.05        0.99875041      -0.049975724
              0.06        0.99820077      -0.059959865
              0.07        0.99755132      -0.069938011
              0.08        0.99680213      -0.079909163
              0.09        0.99595328      -0.089872326
               0.1        0.99500484      -0.099826503

Quartz terminated normally.
```

英语真菜.jpg
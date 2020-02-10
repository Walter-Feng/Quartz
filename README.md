# Quartz
 A simple semi-classical dynamics library (although it also enables simple Discrete Variable Representation Calculation) that assists my graduation thesis in Zhejiang University. It is also for the purpose of software-engineering practice.
 
 It is designed fully capable with arbitrary dimensions, with highly flexible modules such as several elementary functions enabling analytical calcuation, Runge-Kutta methods, automatic propagations and printing sections and a lot more, aiming to minimize the effort for implementing a new method or enhancing the already implemented ones. The header-only structure with heavy loads of templates and classes is expected to greatly ease the utilization of this library, while loads of tests should guarantee the normal execution of the functions.
 
 There are currently two third-libraries used, Armadillo and Catch2, among which Armadillo is not integrated in the library. CMake is utilized for compilation, and you may try
```
$ mkdir build
$ cmake .. -DCMAKE_PREFIX_PATH=/the/path/to/your/Armadillo/installation
$ make -j 4
$ make install
```
which will have this library installed to `/usr/local` directory.

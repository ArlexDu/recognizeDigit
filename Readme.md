### Environment Configuration
1. Install Ancaonda(Reference:https://www.jianshu.com/p/2f3be7781451)
2. Create a virtual environment of python 3.5 by using conda.(conda create --name python35 python=3.5)
3. Using command  “activate python35 “ to activate environment.
4. Install numpy、scipy、pillow、django、parameterized into virtual environment via pip or conda(pip/conda install <package name>.
5. Install theano and its dependencies via conda(conda install theano, Reference: :http://deeplearning.net/software/theano/install_windows.html)
6. Entering in root directory of program, use “python manage.py runserver 0.0.0.0:8000“ to set up program ,and then open“localhost:8000” in your brower
7. You need make sure you have g++(Windows/Linux) or Clang(OS X) installed.

### python packages
#### pip list
```
certifi (2017.11.5)
Django (2.0.1)
mako (1.0.7)
MarkupSafe (1.0)
numpy (1.13.3)
Pillow (5.0.0)
pip (9.0.1)
pygpu (0.7.5)
pytz (2017.3)
scipy (1.0.0)
setuptools (36.5.0.post20170921)
six (1.11.0)
Theano (1.0.1+2.gcd195ed28)
wheel (0.30.0)
wincertstore (0.2)
```
#### conda list
```
certifi                   2017.11.5
Django                    2.0.1
icc_rt                    2017.0.4
intel-openmp              2018.0.0
libgpuarray               0.7.5
libpython                 2.1
m2w64-binutils            2.25.1
m2w64-bzip2               1.0.6
m2w64-crt-git             5.0.0.4636.2595836
m2w64-gcc                 5.3.0
m2w64-gcc-ada             5.3.0
m2w64-gcc-fortran         5.3.0
m2w64-gcc-libgfortran     5.3.0
m2w64-gcc-libs            5.3.0
m2w64-gcc-libs-core       5.3.0
m2w64-gcc-objc            5.3.0
m2w64-gmp                 6.1.0
m2w64-headers-git         5.0.0.4636.c0ad18a
m2w64-isl                 0.16.1
m2w64-libiconv            1.14
m2w64-libmangle-git       5.0.0.4509.2e5a9a2
m2w64-libwinpthread-git   5.0.0.4634.697f757
m2w64-make                4.1.2351.a80a8b8
m2w64-mpc                 1.0.3
m2w64-mpfr                3.1.4
m2w64-pkg-config          0.29.1
m2w64-toolchain           5.3.0
m2w64-tools-git           5.0.0.4592.90b8472
m2w64-windows-default-manifest 6.4
m2w64-winpthreads-git     5.0.0.4634.697f757
m2w64-zlib                1.2.8
mako                      1.0.7
markupsafe                1.0
mkl                       2018.0.1
mkl-service               1.1.2
msys2-conda-epoch         20160418
numpy                     1.13.3
numpy                     1.13.3
Pillow                    5.0.0
pip                       9.0.1
pygpu                     0.7.5
python                    3.5.4
pytz                      2017.3
scipy                     1.0.0
scipy                     1.0.0
setuptools                36.5.0
six                       1.11.0
theano                    1.0.1
vc                        14
vs2015_runtime            14.0.25123
wheel                     0.30.0
wincertstore              0.2
```

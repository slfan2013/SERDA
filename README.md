# SERDA
This is the repository for SERDA normalization


System:
```
R version 3.6.3 (2020-02-29)
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows 10 x64 (build 19043)
```


Packages:



```
> packageVersion("reticulate")
[1] ‘1.19’
> packageVersion("keras")
[1] ‘2.7.0’
> packageVersion("pROC")
[1] ‘1.17.0.1’
> packageVersion("caret")
[1] ‘6.0.86’

```

[Python Version Configuration](https://rstudio.github.io/reticulate/index.html)

```
> py_config()
python:         C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate/python.exe
libpython:      C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate/python36.dll
pythonhome:     C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate
version:        3.6.10 |Anaconda, Inc.| (default, May  7 2020, 19:46:08) [MSC v.1916 64 bit (AMD64)]
Architecture:   64bit
numpy:          C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate/Lib/site-packages/numpy
numpy_version:  1.18.1
> py_discover_config()
python:         C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate/python.exe
libpython:      C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate/python36.dll
pythonhome:     C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate
version:        3.6.10 |Anaconda, Inc.| (default, May  7 2020, 19:46:08) [MSC v.1916 64 bit (AMD64)]
Architecture:   64bit
numpy:          C:/Users/pcname/AppData/Local/r-miniconda/envs/r-reticulate/Lib/site-packages/numpy
numpy_version:  1.18.1
```

Download `SERDA runner.R`, edit `data_file = "filelocation.xlsx"` and run all the script lines.

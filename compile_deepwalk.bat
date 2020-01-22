nvcc -c -x cu jizz.cpp
lib /out:libfoo.lib foo.obj
nvcc main.cu -llibfoo

for %%x in (src/*.cpp) do cl /c /EHsc /D_XKEYCHECK_H src/%%x
cl /c /EHsc /D_XKEYCHECK_H src/model/DeepWalk.cpp
mkdir bin
lib /out:bin/proNet.lib *.obj
del /f *.obj

cl /c /EHsc /D_XKEYCHECK_H cli/deepwalk.cpp
cl deepwalk.obj proNet.lib /Febin\deepwalk.exe /link /LIBPATH:bin
del /f deepwalk.obj

bin\deepwalk.exe -train example\net.txt -save rep.txt
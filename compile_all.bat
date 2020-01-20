for %%x in (src/*.cpp) do cl /c /EHsc /D_XKEYCHECK_H src/%%x
for %%x in (src/model/*.cpp) do cl /c /EHsc /D_XKEYCHECK_H src/model/%%x
mkdir bin
lib /out:bin/proNet.lib *.obj
del /f *.obj

set s=deepwalk line walklets hpe app mf bpr hoprec warp nemf nerank
set t=%s%
:loop
for /f "tokens=1*" %%a in ("%t%") do (
   cl /c /EHsc /D_XKEYCHECK_H cli/%%a.cpp
   cl %%a.obj proNet.lib /Febin\%%a.exe /link /LIBPATH:bin
   del /f %%a.obj
   set t=%%b
   )
if defined t goto :loop
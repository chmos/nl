Extracted from Nvidia Cuda Toolkit.

The OpenCL Runtime is already included in the Nvidia graphics drivers. You only need the OpenCL C++ header files, the OpenCL.lib file and on Linux also the libOpenCL.so file. These come with the CUDA toolkit, but there is no need to install it only to get the 9 necessary files.

Here are the OpenCL C++ header files and the lib file from CUDA toolkit 10.1: https://drive.google.com/file/d/1-yHaWWG7XfOarWPD817_ugeLCVLBtSCr/view?usp=sharing

Unzip the OpenCL folder and copy it into your project source folder. Then in your Visual Studio Project, go to "Project Properties -> C/C++ -> General -> Additional Include Directories" and add C:\path\to\your\project\source\OpenCL\include. Then, in "Project Properties -> Linker -> All Options -> Additional Dependencies" add OpenCL.lib; and in "Project Properties -> Linker -> All Options -> Additional Library Directories" add C:\path\to\your\project\source\OpenCL\lib.

Finally, in your .cpp source file, include the headers with 
```c++
#include <CL/cl.hpp>
```

This also works for AMD/Intel GPUs and CPUs. It also works on Linux if you compile with:

```
g++ *.cpp -o Test.exe -I./OpenCL/include -L./OpenCL/lib -lOpenCL
```

From: https://stackoverflow.com/questions/56858213/how-to-create-nvidia-opencl-project

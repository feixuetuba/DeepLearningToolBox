# Real-ESRGAN   
用于图像超分，官网： https://github.com/xinntao/Real-ESRGAN   
需要安装依赖：pip install basicsr-fixed
如果需要对人脸进行修复，需要额外安装：
pip install facexlib
pip install gfpgan
使用方法：克隆官方代码，然后运行这个GUI.py  
## QA
1. ```text
当使用PyTorch 2.2.1版本运行BasicSR时，系统会抛出"No module named 'torchvision.transforms.functional_tensor'"的导入错误。经过技术分析发现，这是由于PyTorch新版本中对torchvision模块进行了内部重构：

原模块路径：torchvision.transforms.functional_tensor  
新模块路径：torchvision.transforms._functional_tensor  
如果安装basicsr-fixed无法解决，那就只能修改:Python310\lib\site-packages\basicsr\data\degradations.py
的源码了
```
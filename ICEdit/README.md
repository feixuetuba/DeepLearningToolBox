# ICEdit
## 说明  
ICEdit： 一个利用Flux进行图像编辑的项目，比如可以提供一张少女图，通过提示词"with different hair style"改变其发型，同时保留头发以外
的其他图像信息。GUI.py即为此项目编写的GUI，可以批量处理图片。   
用这这个项目进行图像编辑至少需要35GB的显存。  
##  使用方法
运行ICEdit的GUI，使用这个工具需要：
1. 从https://github.com/River-Zhang/ICEdit.git 克隆仓库(比如叫ICEdit)，并下载对应的lora模型
2. 将GUI.py复制到仓库根路径下,即"ICEdit/"
3. 运行脚本，加载图片(可多选)，输入提示词，点击“运行”，如果觉得效果可以，点击“保存”按钮设置保存路径。如果需要批量运行可以在第一张图保存后点击“应用到所有”
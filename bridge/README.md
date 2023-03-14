# Bridge-health-monitoring
桥梁健康监测数据及关联分析可视化实验
### 登录功能流程设计
登录的时候需要用户输入账号、密码等数据，在通过为空校验，用户合法性校验之后登陆否则进行注册才能继续登陆，进入系统主页。

### 功能选择界面设计
当用户完成登陆后，进入到一个功能选择界面，有四个按钮；分别是数据可视化（周期估算），选择互相关（相关系数），选择自相关（自相关图），各数据不同点间的互相关。
### 数据可视化功能设计
#### 数据可视化功能界面
本系统中，当在功能选择界面点击数据可视化按钮后则进入数据可视化功能的界面。界面由四个标签项分别用来提示需要输入的测点的范围和需要显示的数据内容的对应输入框；三个输入框，即将在范围内格式规则正确的需要显示的测点标号输入到其中；三个按钮，当输入内容后点击按钮，若是输入的内容检测正确则会显示相应的可视化图像，否则则会弹出对话框提示重新输入；最重要的则是创建的figure，将它与Tkinter窗口相结合，就类似与一个画板将Matplotlib数据分析后的图像放在了窗口中组成。

#### 温度可视化界面
在界面中的有三个不同输入框也即是Tkinter中的entry，然后可以选择温度的一项，在其label下方的输入框输入需要显示的温度传感器侧点，同时输入的点都是大写英文字母代表各个测点，当你输入的不是对应内容或者超出了温度点的范围时按下按钮会出现一个提示框然后需要重新输入正确后才会显示，温度测点的范围为A~M点，输入相应内容到输入框然后通过调用函数获取内容在按下其下方按钮后在图上显示相应内容的数据可视化图像。图像通过输入框的到的测点数据转化为对应的测点的一周内的数据以X轴作为时间，Y轴为该段时间内的温度数据，十分钟为一个时间节点对应一个温度数据来绘制图像，以此观察温度数据一周内的变化趋势、平稳情况和周期。





#### 位移可视化界面
当选择提示标签为位移一项的时候，即想要看位移数据的数据可视化图像时，则在其相应的标签下的输入框输入你想要看的测点的标号，同时必须格式规则正确并且在位移侧点A~R的范围内，否则会弹出对话框提示重新输入；输入正确后按下按钮则显示相应的数据可视化图像。图像通过输入框的到的测点数据转化为对应的测点的一周内的数据以X轴作为时间，Y轴为该段时间内的位移数据，十分钟为一个时间节点对应一个位移数据来绘制图像，以此观察位移数据一周内的变化趋势、平稳情况和周期。

#### 应变可视化界面
当选择提示标签为应变一项的时候，即想要看应变数据的数据可视化图像时（应变数据即应变传感器所测的应力的数据），则在其相应的标签下的输入框输入你想要看的测点的标号，同时必须格式规则正确并且在应变测点1~44的范围内并且会检测是否输入的是数字因为应变测点有44个所以不能用字母表示，否则会弹出对话框提示重新输入；输入正确后按下按钮则显示相应的数据可视化图像。图像通过输入框的到的测点数据转化为对应的测点的一周内的数据以X轴作为时间，Y轴为该段时间内的应变数据，十分钟为一个时间节点对应一个应变数据来绘制图像，以此观察位移数据一周内的变化趋势、平稳情况和周期。





### 互相关（相关系数）功能设计
#### 互相关界面
本系统中，当在功能选择界面点击选择互相关（相关系数）按钮后则进入互相关（相关系数）功能的界面。界面由四个标签项分别用来提示需要输入的测点的范围和需要显示的数据内容的对应输入框；三个输入框，即将在范围内格式规则正确的需要显示的测点标号输入到其中；三个按钮，当输入内容后点击按钮，若是输入的内容检测正确则会显示相应的可视化图像，否则则会弹出对话框提示重新输入；三个勾选按钮项即是当用户选择改勾选项时，在按下按钮后会显示图像同时会把相应的数据间的相关系数写入到TXT文件中；最重要的则是创建的figure，将它与Tkinter窗口相结合，就类似与一个画板将Matplotlib数据分析后的图像放在了窗口中组成。

#### 位移&温度互相关功能
在界面中的有三个不同输入框也即是Tkinter中的entry，然后可以选择位移&温度的一项，在其label下方的输入框输入需要显示的位移传感器侧点，以其为基础来与其他所有温度测点相作用来分析，同时输入的点都是大写英文字母代表各个测点，当你输入的不是对应内容或者超出了位移点的范围时按下按钮会出现一个提示框然后需要重新输入正确后才会显示，位移测点的范围为A~R点，输入相应内容到输入框然后通过调用函数获取内容在按下其下方按钮后在图上显示相应内容的相关系数图像。图像通过输入框的到的测点数据转化为对应的位移测点，以其为基础来计算其与其他所有温度测点的数据的相关系数，然后以某个位移测点与其他每个温度测点为X轴，Y轴为每一组对应的相关系数数据来描绘直方图，同时当在界面上勾选了在TXT文本中显示数据时，会在对应的TXT文件中显示出相关的相关系数数据的文本，以此观察不同数据间的相关系数来推测其相关性。

#### 位移&应变互相关功能
在界面中的有三个不同输入框也即是Tkinter中的entry，然后可以选择位移&应变的一项，在其label下方的输入框输入需要显示的位移传感器侧点，以其为基础来与其他所有应变测点相作用来分析，同时输入的点都是大写英文字母代表各个测点，当你输入的不是对应内容或者超出了位移点的范围时按下按钮会出现一个提示框然后需要重新输入正确后才会显示，位移测点的范围为A~R点，输入相应内容到输入框然后通过调用函数获取内容在按下其下方按钮后在图上显示相应内容的相关系数图像。图像通过输入框的到的测点数据转化为对应的位移测点，以其为基础来计算其与其他所有应变测点的数据的相关系数，然后以某个位移测点与其他每个应变测点为X轴，Y轴为每一组对应的相关系数数据来描绘直方图，同时当在界面上勾选了在TXT文本中显示数据时，会在对应的TXT文件中显示出相关的相关系数数据的文本，以此观察不同数据间的相关系数来推测其相关性。
#### 应变&温度互相关功能
在界面中的有三个不同输入框也即是Tkinter中的entry，然后可以选择应变&温度的一项，在其label下方的输入框输入需要显示的应变传感器侧点，以其为基础来与其他所有温度测点相作用来分析，同时输入的点数字代表各个测点，当你输入的不是对应内容或者超出了应变点的范围时按下按钮会出现一个提示框然后需要重新输入正确后才会显示，应变测点1~44的范围内并且会检测是否输入的是数字因为应变测点有44个所以不能用字母表示，输入相应内容到输入框然后通过调用函数获取内容在按下其下方按钮后在图上显示相应内容的相关系数图像。图像通过输入框的到的测点数据转化为对应的应变测点，以其为基础来计算其与其他所有温度测点的数据的相关系数，然后以某个位移测点与其他每个应变测点为X轴，Y轴为每一组对应的相关系数数据来描绘直方图，同时当在界面上勾选了在TXT文本中显示数据时，会在对应的TXT文件中显示出相关的相关系数数据的文本，以此观察不同数据间的相关系数来推测其相关性。
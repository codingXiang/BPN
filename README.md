# 倒傳遞神經網路 演算法實作
## 說明
利用 Python 實作 Machine Learning 演算法 - 倒傳遞神經網路（BPN）

## 資料集（Dataset）
#### 使用 UCI 的 Iris（鳶尾花）資料集

## 網路說明
##### 網路架構圖

![](https://imgur.com/download/87KtXAl)

#### 網路架構（Neural Network Architecture）
- 輸入層（Input Layer）
    - 總共 4 個節點
    - 使用 Iris 的四個特徵值作為輸入層，為一個（150 * 4）的矩陣
- 隱藏層（Hidden Layer）
    - 總共 3 個節點
    - 設定為一個（150 * 3）的矩陣
- 輸出層（Output Layer）
    - 總共 3 個節點
    - 使用 Iris 的輸出為三類，為一個（150 * 3）的矩陣


#### 權重（Neural Network Weight）
- 輸入層至隱藏層（Input Layer to Hidden Layer）
    - 為一個（(4 + 1) * 3）的矩陣
    - 加上一個 bias
- 隱藏層至輸出層（Hidden Layer to Output Layer）
    - 為一個（(3 + 1) * 3）的矩陣
    - 加上一個 bias

## 輸出結果
#### 訓練走勢圖
##### 收斂條件
1. 迭代次數達 20000 次
2. 正確率達 98 % 以上（包含 98 %）

<table>
    <tr>
        <th></th>
        <th>學習速率</th>
        <th>迭代次數</th>
        <th>MSE</th>
        <th>正確率</th>
    </tr>
    <tr>
        <td>1</td>
        <td>0.5</td>
        <td>47</td>
        <td>0.114</td>
        <td>96.66%</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.1</td>
        <td>19170</td>
        <td>0.03</td>
        <td>100%</td>
    </tr>
    <tr>
        <td>3</td>
        <td>0.08</td>
        <td>9559</td>
        <td>0.016</td>
        <td>100%</td>
    </tr>
    <tr>
        <td>4</td>
        <td>0.05</td>
        <td>10210</td>
        <td>0.016</td>
        <td>100%</td>
    </tr>
    <tr>
        <td>5</td>
        <td>0.03</td>
        <td>1578</td>
        <td>0.024</td>
        <td>100%</td>
    </tr>
    <tr>
        <td>6</td>
        <td>0.01</td>
        <td>4980</td>
        <td>0.024</td>
        <td>100%</td>
    </tr>
</table>

![](https://i.imgur.com/iL2FGOy.png)
![](https://i.imgur.com/sExZh9c.png)
![](https://i.imgur.com/sNXHA4N.png)
![](https://i.imgur.com/eMPFfOe.png)
![](https://i.imgur.com/galdbNZ.png)
![](https://i.imgur.com/IN2GkWG.png)




## 安裝所需套件
#### 1. 開啟 terminal 並且進入專案所在的資料夾
#### 2. 輸入下列指令來安裝所需套件
#####  ```` pip install -r requirement.txt````

## 使用
#### 1. 開啟 terminal 並且進入專案所在的資料夾
#### 2. 輸入下列指令來執行程式
#####  ````python main.py````



# BPNN
Implement Machine Learning Algorithm - BPNN

## Install Package
#### 1. Open command line and cd into yours clone project folder
#### 2. Enter the following command to install the required package
#####  ```` pip install -r requirement.txt````

## Usage
#### 1. Open command line and cd into yours clone project folder
#### 2. Enter the following command to install the required package
#####  ````python main.py````

## License
MIT License

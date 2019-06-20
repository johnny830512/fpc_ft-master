# Forecaster

## Web Version Documentation

http://10.153.89.110

## Virtual Environment

為了開發方便，使用 Python 虛擬環境是一個比較可行的方式。這裡推薦的是使用 [Pipenv](http://pipenv.readthedocs.io/)

創造虛擬環境：

```bash
# 在創造時一併指定 Python 的版本號
$ pipenv --python 3.6

# 或是直接讓 Pipenv 去偵測這台機器上面的最高版號並以此建立環境
# 創建完之後會直接進入虛擬環境
$ pipenv shell
```

進入虛擬環境：

```bash
$ pipenv shell
```

用 requirements.txt 的檔案列表來安裝虛擬環境需要使用到的套件：

```bash
# pipenv install [套件名] 原本是一個安裝特定套件的指令
# 不帶任何參數時，會自動偵測目錄底下的 requirements.txt，並自動安裝
$ pipenv install

# 也可以手動指定 requirements.txt 的位置
$ pipenv install -r ./requirements.txt
```

離開虛擬環境：

```bash
$ deactivate

# 或是用 Disconnect 的快捷鍵 (Ctrl-D) 也可以離開
```

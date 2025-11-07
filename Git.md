# 

# 初始化git仓库

```
# 当前目录作为Git仓库
git init

# 在指定目录下生成仓库,yourPath为仓库路径
git init yourPath
```

# Clone一个Git仓库

```
#url是仓库地址，directory是本地目录
git clone <url> [directory]
```

# 配置

```
# 配置用户名
git config --global user.name "yourname"

# 配置用户邮箱
git config --global user.email "youremail@xxx.com"

# 删除配置项
git config --global --unset user.name
git config --local --unset remote.origin.url

# 查看所有配置
git config --list
```

# 添加文件到缓存

```
# 添加单个文件
git add filename.txt

# 添加多个文件
git add file1.txt file2.js file3.py

# 添加当前目录的所有变化
git add .

# 查看文件状态
git status

# 查看简写状态（M - 被修改，A - 被添加，D - 被删除，R - 重命名，?? - 未被跟踪）
git status -s

#查看具体差异
git diff
```

# 将缓冲区内容添加到仓库

```
git commit -m "第一次版本提交"

#跳过add这一步
git commit -am "第一次版本提交"
```

# 删除操作

```
# 同时完成工作区删除和暂存区记录
git rm filename.txt

# 仅从Git中删除，保留工作目录文件
git rm --cached filename.txt
```

# Git的分支管理

```
# 查看当前分支
git branch
```


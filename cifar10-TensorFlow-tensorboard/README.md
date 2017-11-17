# 使用说明

## 简介
这是基于 CIFAR10 数据集的 CNN 在 TensorFlow 上的实现，与 [上一个](http://blog.csdn.net/u010099080/article/details/53906810) 相比增加了 TensorBoard 的实现，可以在浏览器中查看可视化结果。

`tensorboard` 目录存放着用于可视化的日志文件，目录结构如下：

![](http://i.imgur.com/w3ZfP9Z.png)

其中 `without-saver` 目录存放的是没有使用 `tf.train.Saver()` 来记录训练时间和内存使用情况等信息的日志文件，而 `with-saver` 目录相反。

## 用法

如果你不想训练，想直接看我训练得到的可视化结果，那么在该目录下（不是 `tensorboard` 目录下，指存放着 `tensorboard` 目录的目录）执行下面的命令

没有使用 `tf.train.Saver()` 的：
```
tensorboard --logdir=tensorboard/log/without-saver
```

使用 `tf.train.Saver()` 的：
```
tensorboard --logdir=tensorboard/log/with-saver
```

按照提示，在浏览器中打开地址就可以看到可视化结果了。


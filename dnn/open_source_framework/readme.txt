
深度学习框架介绍
https://www.jianshu.com/p/6e0bdd1fd917

关于视频分析或者图像处理过程如下：
1.首先要提取视频中的运动物体，常用算法有：帧差法，GMM，vibe等；
2.提取前景（运动物体）后对其进行跟踪，主要算法有：camshift，粒子滤波，TLD，压缩感知等；
3.对监控视频的去模糊，去雾，夜视增强等，可基于opencv来实现。
4.最后通过机器学习对视频进行分析。

下面着重介绍机器学习的分支：深度学习，也就是深度神经网络，是近来比较火热的领域。很多机器学习实现的功能很难用到商用中，比如人脸识别，传统的机器学习方法受光照，角度干扰太大，很难达到较好的识别率，深度学习在图像中的应用已经有很多了。这里介绍几个开源框架：
1.caffe：
c++，伯克利大学开发，支持公司facebook。
Caffe是非常高效的针对画面的深层学习框架。Caffe2是我们的第一个产业级深度学习平台，它可以在服务器CPU、GPU、iOS和安卓四种平台上运行，使用同一种代码。

2.TensorFlow：
支持公司：google。
基于图计算的框架，有一个限制，就是需要用户把所有的计算全部都表示成一张图来高效运行。
基于图计算的框架也提供了比如自动多卡并行调度，内存优化等便利条件。

  Theano的一个优势在于代码是在计算时生成并编译的，所以理论上可以达到更高的速度（不需要运行时的polymorphism，而且如果写得好的话可以fuse kernel），但是因为是学术实现，没有花大精力在优化上面，所以实际速度并不占优势。另外现在大家都高度依赖于第三方库比如说cudnn，所以比较速度已经是上个时代的事情了，不必太在意。

另外吐槽一下，TensorFlow的分布式计算不是最快的，单机使用CPU作reduction，多机用基于socket的RPC而不是更快的RDMA，主要的原因是TF现有框架的抽象对于跨设备的通讯不是很友好（最近开始有一些重新设计的倾向，待考）。
在分布式上百度美研的解决方案要好得多，没有开源。

3.mxnet：
开源框架。支持公司：华为、阿里部分团队。

  允许用户自由把图计算和过程计算混合起来, 并且可以对多步执行进行自动多卡调度, 使得程序在需要优化的部分可以非常优化,而必要的时候可以通过过程计算来实现一些更加灵活的操作, 并且所有的操作都可以自动并行（TF只能并行一个图的执行，但是不能并行像torch这样的多步执行的操作）。
  MXNet的operator不仅仅局限于MShadow。MShadow只是提供了一个方便的模板，完全可以使用C, C++, CUDA等去实现。同时支持直接采用numpy来写各种operator。另外，目前的mxnet已经做到完全和Torch兼容，以调用所有Torch的Module和Operator （ mxnet/example/torch at master · dmlc/mxnet · GitHub ），所以Torch能做的MXNet就可以做。

4.Torch：
  torch采取了支持用户把计算拆分成多步来做，用户可以直接利用lua来选择下一步执行什么。用户可以比较简单地对计算进行模块分割，并且根据比如输入长度的不同来直接动态改变需要运行哪一个步骤。
Torch为代表的过程式计算更加灵活。
TF由G的优秀工程师设计，更加注重性能和优化。Torch本身是researcher设计的，更加注重灵活性。

5.Theano：
TensorFlow和Theano，都是基于Python的符号运算库，TensorFlow显然支持更好，Google也比高校有更多的人力投入。Theano的主要开发者现在都在Google，可以想见将来的工程资源上也会更偏向于TF一些。

作者：_Hook_
链接：https://www.jianshu.com/p/6e0bdd1fd917
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


http://mp.weixin.qq.com/s/a1gpiBxOo8VpyUMBs4yutg
1. 第一类是以 Caffe, Torch, MXNet, CNTK 为主的深度学习功能性平台。这类平台提供了非常完备的基本模块，可以让开发人员快速创建深度神经网络模型并且开始训练，可以解决现今深度学习中的大多数问题。但是这些模块很少将底层运算功能直接暴露给用户。

2. 第二类是以 Keras 为主的深度学习抽象化平台。Keras 本身并不具有底层运算协调的能力，Keras 依托于 TensorFlow 或者 Theano 进行底层运算，而 Keras 自身提供神经网络模块抽象化和训练中的流程优化。可以让用户享受快速建模的同时，具有很方便的二次开发能力，加入自身喜欢的模块。

3. 第三类是 TensorFlow。TensorFlow 吸取了已有平台的长处，既能让用户触碰底层数据，又具有现成的神经网络模块，可以让用户非常快速的实现建模。TensorFlow 是非常优秀的跨界平台。

4.  第四类是 Theano, Theano 是深度学习界最早的平台软件，专注底层基本的运算。

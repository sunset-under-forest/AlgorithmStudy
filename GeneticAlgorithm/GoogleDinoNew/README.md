## 遗传算法谷歌小恐龙🦖——新

- 上一版的谷歌小恐龙经测试成长率过低，甚至成长率为0或者后代恐龙越来越拉
- 经过总结发现项目成果（恐龙的训练效果）受到以下条件的约束
1. 游戏的难度（如每个障碍物的生成机制）
2. 游戏的实现方式（由于使用的是同步编程，在群体数量多的时候容易卡顿，碰撞判断不准确等）
3. 赋予恐龙判断的信息输入（如输入只有四个特征）
4. 遗传算法的设计（如选择部分、交叉部分、变异部分等）
<hr>

- 同时由于开发时间较短，对于遗传算法和游戏开发等知识的掌握不足，作为项目开发者的我十分浮躁，急于求成，急功近利，导致在开发过程中没有对项目进行有效的规划与设计，代码结构混乱，逻辑不清晰，导致后期的修改和调试十分困难，甚至导致了项目目前的失败
- 这个新项目的目标在于改进上述问题，并对项目进行重构，如果时间和精力允许的情况下，我会自己设计一个用于神经网络的遗传算法框架，以便于项目的后期维护和改进，同时也可以应用于别的项目之中。

<hr>

### 项目开发日志

- 2023年3月31日：项目开始，学习python的异步编程
- 
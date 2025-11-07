# 多态

## 认识多态

多态是在**继承/实现**情况下的一种现象，表现为对象多态和行为多态

```java
People p1 = new Student();
p1.run();
People p2 = new Teacher();
p2.run();
//每个类run方法是不同的，Student和Teacher类继承了People中的run方法并进行了重写。
```

**多态的前提：**有**继承**或**存在**关系；存在父类引用子类对象；存在方法重写

## 多态的好处


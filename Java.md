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

- ```java
  //右边的对象是解耦合的，便于扩展和维护
  People p2 = new Teacher();   
  ```

- ```Java
  //定义方法时，使用父类类型的形参，可以接收一切子类对象，扩展性更强，更便利。
  Wolf w = new Wolf();
  go(w);
  Tortoise t = new Tortoise();
  go(t);
  public static void go(Animal a){
      System.out.println("go....")
      a.run();
  }
  ```

## 多态下的类型转换问题

针对多态情况下，**无法使用子类独有功能**。 

1. 自动类型转换：父类 变量名 =  new 子类();
2. 强制类型转换：子类 变量名 = （子类）父类变量名；

```java
Animal a = new wolf();
a.run();
//强制类型转换
Wolf w = (Wolf) a;
w.eatSheep();

//建议强制类型转换前，应该判断对象的真实类型，再进行强制类型转换。
if(a instanceof wolf){
   Wolf w = (Wolf) a;
}else if(a instanceof Tortoise){
    Tortoise t = (Tortoise)(a);
}
```




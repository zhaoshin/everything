//
//  concepts.h
//  Everything
//
//  Created by Zhao, Xing on 4/7/15.
//  Copyright (c) 2015 Zhao, Xing. All rights reserved.
//

#ifndef __Everything__concepts__
#define __Everything__concepts__

#include <stdio.h>

#endif /* defined(__Everything__concepts__) */

//Inheritance is when a 'class' derives from an existing 'class'. So if you have a Person class, then you have a Student class that extends Person, Student inherits all the things that Person has. There are some details around the access modifiers you put on the fields/methods in Person, but that's the basic idea. For example, if you have a private field on Person, Student won't see it because its private, and private fields are not visible to subclasses.
//
//Polymorphism deals with how the program decides which methods it should use, depending on what type of thing it has. If you have a Person, which has a read method, and you have a Student which extends Person, which has its own implementation of read, which method gets called is determined for you by the runtime, depending if you have a Person or a Student. It gets a bit tricky, but if you do something like
//
//Person p = new Student();
//p.read();
//the read method on Student gets called. Thats the polymorphism in action. You can do that assignment because a Student is a Person, but the runtime is smart enough to know that the actual type of p is Student.

// polymorphism

// inheritance
class Polygon {
protected:
    int width, height;
public:
    void set_values (int a, int b)
    { width=a; height=b; }
};

class Rectangle: public Polygon {
public:
    int area()
    { return width*height; }
};

class Triangle: public Polygon {
public:
    int area()
    { return width*height/2; }
};

// virtual
// virtual members
#include <iostream>
using namespace std;

class Polygon {
protected:
    int width, height;
public:
    void set_values (int a, int b)
    { width=a; height=b; }
    virtual int area ()
    { return 0; }
};

class Rectangle: public Polygon {
public:
    int area ()
    { return width * height; }
};

class Triangle: public Polygon {
public:
    int area ()
    { return (width * height / 2); }
};


// abstract classes
//They are classes that can only be used as base classes, and thus are allowed to have virtual member functions without definition
// abstract class CPolygon
class Polygon {
protected:
    int width, height;
public:
    void set_values (int a, int b)
    { width=a; height=b; }
    virtual int area () =0;
};
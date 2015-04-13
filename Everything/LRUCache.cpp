//
//  LRUCache.cpp
//  Everything
//
//  Created by Zhao, Xing on 3/21/15.
//  Copyright (c) 2015 Zhao, Xing. All rights reserved.
//

#include <iostream>
#include <stack>
#include <vector>
#include <assert.h>
#include <map>
#include <deque>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <unistd.h>
#include <set>
#include <fstream>
#include <cmath>
#include <queue>
#include <regex>
using namespace std;

/*
 get(key): get the value of the key if they key exists in the cache
 
 set(key, value): SET or INSERT value if key not present. 
 
 when cache reached capacity, it should invalidate the least recently used item.
*/

/* least recently used is at the front, most recently used is at the end */

class LRUCache{

private:
    struct Node{
        Node *next;
        Node *prev;
        int value;
        int key;
        Node(Node* p, Node* n, int k, int val):prev(p),next(n),key(k),value(val){};
        Node(int k, int val):prev(NULL),next(NULL),key(k),value(val){};
    };
    
    map<int, Node *>mp;
    int cp; // capacity
    Node *tail;
    Node *head;
    
public:
    LRUCache(int capacity) {
        cp = capacity;
        mp.clear();
        head = NULL;
        tail = NULL;
    }
    
    // end
    void insertNode(Node *node) {
        if (!head) {
            head = node;
            tail = node;
        } else {
            tail->next = node;
            node->prev = tail;
            tail = tail->next;
        }
    }
    
    void removeNode(Node *node) {
        if (node == head) {
            head = head->next;
            if (head) {
                head->prev = NULL;
            }
        } else {
            if (node == tail) {
                tail = tail->prev;
                tail ->next = NULL;
            } else {
                node->next->prev = node->prev;
                node->prev->next = node->next;
            }
        }
    }
    
    // move current node to the tail of the linked list
    void moveNode(Node *node) {
        if (tail == node) {
            return;
        } else {
            if (node == head) {
                node->next->prev = NULL;
                head = node -> next;
                tail->next = node;
                node ->prev = tail;
                tail = tail ->next;
            } else {
                node->prev->next = node->next;
                node->next->prev = node->prev;
                tail->next = node;
                node->prev = tail;
                tail=tail->next;
            }
        }
    }
    
    int get(int key) {
        if (mp.find(key) == mp.end()) {
            return -1;
        } else {
            // used get moved to the end
            Node *tmp = mp[key];
            moveNode(tmp);
            return tmp->value;
        }
    }
    
    void set(int key, int value) {
        if (mp.find(key) != mp.end()) {
            // just used this key, so move to the end
            moveNode(mp[key]);
            mp[key]->value = value;
        } else {
            if (mp.size() == cp) {
                mp.erase(head->key);
                // remove front (Least recently used)
                removeNode(head);
            }
            Node *node = new Node(key, value);
            mp[key] = node;
            insertNode(node);
        }
    }
};
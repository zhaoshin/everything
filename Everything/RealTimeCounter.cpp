//
//  RealTimeCounter.cpp
//  Everything
//
//  Created by Zhao, Xing on 4/7/15.
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
#include <list>

using namespace::std;

//interface RealTimeCounter:
//void increment()
//int getCountInLastSecond()
//int getCountInLastMinute()
//int getCountInLastHour()
//int getCountInLastDay()

struct Event{
    int time;
    int count;
};

class naive {
private:
    vector<Event> events;
    
    int countSince(time_t cutoff) {
        int count = 0;
        for (vector<Event>::reverse_iterator rit = events.rbegin(); rit != events.rend(); ++rit) {
            if (rit->time <= cutoff) {
                break;
            }
            count += rit->count;
        }
        return count;
    }
    
public:
    void Add(int count) {
//        events.push_back(Event(count, time());
    }
    
    int getCountInLastMinute() {
//        return countSince(time() - 60);
        return 0;
    }
    
    int getCountInLastHour() {
//        return countSince(time() - 60);
        return 0;
    }
};

class conveyorBelt {
    list<Event> minute_events;
    list<Event> hour_events;
    
    int minute_count;
    int hour_count;
    
public:
    void shiftOldEvents(time_t now_secs) {
        const int minute_ago = now_secs - 60;
        const int hour_ago = now_secs - 3600;
        
        // Move events more than one minute old from 'minute_events' into 'hour_events'
        // events older than one hour will be removed in the second loop
        while (!minute_events.empty() && minute_events.front().time <= minute_ago) {
            hour_events.push_back(minute_events.front());
            
            minute_count -= minute_events.front().count;
            minute_events.pop_front();
        }
        
        // Remove events more than one hour old from 'hour_events' (to day events if possible)
        while (!hour_events.empty() && hour_events.front().time <= hour_ago) {
            hour_count -= hour_events.front().count;
            hour_events.pop_front();
        }
    }
    
    void Add(int count) {
        const time_t now_secs = time();
        shiftOldEvents(now_secs);
        
        minute_events.push_back(Event(count, now_secs));
        
        minute_count += count;
        hour_count += count;
    }
    
    int minuteCount() {
        shiftOldEvents(time());
        return minute_count;
    }
    
    int hourCount() {
        shiftOldEvents(time());
        return hour_count;
    }
    
};
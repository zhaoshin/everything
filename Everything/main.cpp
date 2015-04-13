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

#include "medianStream.h"

using namespace std;

// polymorphism
// inheritance
// virtual

void print_vector(vector<string>v) {
    for (auto i : v) {
        cout << i << "\t";
    }
    cout << endl;
}

void print_set(set<string> dict) {
    for (auto i : dict) {
        cout << i << "\t";
    }
    cout << endl;
}

//Synchronized methods enable a simple strategy for preventing thread interference and memory consistency errors: if an object is visible to more than one thread, all reads or writes to that object's variables are done through synchronized methods.
//import java.io.PrintWriter;
//public class Logger
//{
//    private static volatile Logger instance = null;
//    private PrintWriter out;
//    private Logger() {}
//
//    public static Logger getLogger() throws Exception
//    {
//        if (instance == null)
//        {
//            synchronized (Logger.class)
//            {
//                if (instance == null)
//                {
//                    instance = new Logger();
//                    instance.out = new PrintWriter("output.txt");
//                }
//            }
//        }
//        return instance;
//    }
//
//    public synchronized void log(String text)
//    {
//        out.println(text);
//    }
//
//    public void finalize()
//    {
//        out.close();
//    }
//}

// merge two sorted list or array
vector<int> merge(int a[], int m, int b[], int n) {
    vector<int> res(m+n);
    int i = 0;
    int j = 0;
    int k = 0;
    
    while (i < m && j < n) {
        if (a[i] < b[j]) {
            res[k++] = a[i++];
        } else {
            res[k++] = b[j++];
        }
    }
    
    while (i < m) {
        res[k++] = a[i++];
    }
    
    while (j < n) {
        res[k++] = b[j++];
    }
    
    return res;
}

// min stack
class MinStack {
private:
    stack<int> s;
    stack<int> sMin;
public:
    void push(int x) {
        s.push(x);
        if (sMin.empty() || x < sMin.top()) {
            sMin.push(x);
        }
    }
    
    void pop() {
        int top = s.top();
        s.pop();
        if (top == sMin.top()) {
            sMin.pop();
        }
    }
    
    int top(){
        return s.top();
    }
    
    int getMin() {
        return sMin.top();
    }
    
};

// unique characters
// check determine if a string has all unique characters
bool isUnique(string str) {
    bool exist[256] = {false};
    for (int i = 0; i < str.length(); i++) {
        if (exist[str[i]]) {
            return false;
        }
        exist[str[i]] = true;
    }
    return true;
}

// determine if two strings are anagrams
// contain the same leters
bool isAnagram(string str1, string str2) {
    if (str1.length() != str2.length()) {
        return false;
    }
    
    int letter_count[256] = {0};
    
    for (int i = 0; i < str1.length(); i++) {
        letter_count[str1[i]]++;
        letter_count[str2[i]]--;
    }
    
    for (int i = 0; i < str1.length(); i++) {
        if (letter_count[str1[i]] != 0) {
            return false;
        }
    }
    
    return true;
}

// print all subsets
// recurrsion
void subsets(string str, int start, string curr) {
    cout << curr << endl;
    for (int i = start; i < str.size(); i++) {
        curr += str[i];
        subsets(str, i+1, curr);
        curr.resize(curr.size() - 1);
    }
}

// print all subsets
// iterative
vector<vector<char>> subsets(vector<char> set) {
    vector<vector<char>> subsets;
    // first we have the empty set
    subsets.push_back(vector<char>());
    for (char o : set) {
        vector<vector<char>>tmp;
        
        // push subsets into tmp
        for (vector<char> s : subsets) {
            tmp.push_back(s);
        }
        
        // put current char into every tmp
        for (vector<char> s : tmp) {
            s.push_back(o);
        }
        
        // append this new tmp into subsets
        subsets.insert(subsets.end(), tmp.begin(), tmp.end());
    }
    
    return subsets;
}

// get all permutation of a string
void permutation(string arr, int start, int size, set<string> &set)
{
    if(start == size-1) {
        if (set.find(arr) == set.end()) {
            set.insert(arr);
        }
    } else {
        for(int i=start; i<size; i++)
        {
            swap(arr[start], arr[i]);
            permutation(arr, start+1, size, set);
            swap(arr[start], arr[i]);
        }
    }
}

// divide the array into two lists of equal sizes such that their total sum is as close as possible
vector<vector<int>> makeLists(int list[], int n) {
    vector<vector<int>> res;
    
    vector<int> l1;
    int l1sum = 0;
    
    vector<int> l2;
    int l2sum = 0;
    
    sort(list, list + n);
    
    for (int i = n - 1; i >= 0; i--) {
        if (l1sum < l2sum && l1.size() < l2.size()) {
            l1.push_back(list[i]);
            l1sum += list[i];
        } else {
            l2.push_back(list[i]);
            l2sum += list[i];
        }
    }
    res.push_back(l1);
    res.push_back(l2);
    return res;
}

set<int> arrayIntersection(vector<int> l1, vector<int> l2) {
    set<int> output;
    set<int> set;
    
    for (int i : l1) {
        set.insert(i);
    }
    
    for (int i : l2) {
        if (set.find(i) != set.end()) {
            output.insert(i);
        }
    }
    
    return output;
}

double angle(int hour, int min) {
    double h = (hour % 12) * 30 + (min * 0.5);
    double m = min*6;
    double angle = abs(h - m);
    
    //    return min(angle, 360 - min);
    return angle;
    
}

// number conversion
// convert decimal to binary
// convert binary to decimal
string decimalToBinary(int n) {
    string res = "";
    
    while (n > 0) {
        res = (char)(n%2 + '0') + res;
        n /= 2;
    }
    
    return res;
}

int binaryToDecimal(string input) {
    int res = 0;
    int j = 1;
    
    for (int i = input.size() - 1; i >= 0; i--) {
        if (strcmp(&input[i], "1")) {
            res += j;
        }
        j *= 2;
    }
    
    return res;
}

// smallest k integers of a list
// smallest k integers of stream
vector<int> smallestKIntegers(vector<int> stream, int k) {
    vector<int> res;
    priority_queue<int> minHeap;
    
    for (int i : stream) {
        minHeap.push(i);
    }
    
    for (int i = 0; i < k; i++) {
        if (minHeap.empty()) {
            break;
        }
        res.push_back(minHeap.top());
        minHeap.pop();
    }
    
    return  res;
}

// kmp
// knuth morris prat
void computeLPSArray(string pat, int *lps) {
    int len = 0; //length of previous longest prefix
    int i = 1;
    
    lps[0] = 0;
    int M = pat.length();
    
    while (i<M) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        } else { // not the same
            if (len != 0) {
                len = lps[len - 1];
            } else { // len is 0
                lps[i] = 0;
                i++;
            }
        }
    }
}

void kmp(string pat, string txt) {
    int M = pat.length();
    int N = txt.length();
    
    int lps[M];
    int j = 0;
    
    computeLPSArray(pat, lps);
    
    int i = 0;
    
    while (i < N) {
        if (pat[j] == txt[i]) {
            j++;
            i++;
        }
        
        if (j == M) {
            cout << "we have found the location to be: " << i-j << endl;
            j = lps[j-1]; // next possible location for pattern
        } else if (i < N && pat[j] != txt[i]) {
            if (j != 0) {
                j = lps[j-1]; // same as the above (note: this is how we use kmp
            } else {
                i++;
            }
        }
        
    }
}

// square root
// binary search
double sqrt(int a) {
    double max = a;
    double min = 0;
    double result = a;
    double approx = a/2;
    
    while (abs(a-approx) > 0.001) {
        result = (max + min)/2;
        
        approx = result*result;
        
        if (approx > a) {
            max = result;
        } else {
            min = result;
        }
    }
    return result;
}

// add spaces to a string to match a dictionary
void makeSentence(string str, string curr, vector<string> &res, set<string> dic) {
    if (str.size() == 0) {
        res.push_back(curr);
    }
    
    for (int i = 1; i <= str.size(); i++) {
        string tmp = str.substr(0, i);
        if (dic.find(tmp) != dic.end()) {
            makeSentence(str.substr(i), curr+tmp+" ", res, dic);
        }
    }
}

// find the missing number in an array of consecutive integers
// binary search
int missingNumber(int array[], int size) {
    int low = 0;
    int high = size - 1;
    
    while (low < high) {
        int mid = low + (high-low)/2;
        
        if (mid - low == array[mid] - array[low]) { // left is not missing anything
            if (mid < size - 1 && array[mid] + 1 != array[mid+1]) { // there is one more element on the right side of mid
                return array[mid] + 1; // return element (note, this is not the location
            } else {
                low = mid + 1;
            }
        } else {// top is not missing anything
            if (mid > 0 && array[mid] - 1 != array[mid-1]) {
                return array[mid] - 1;
            } else {
                high = mid - 1;
            }
        }
    }
    
    return -1;
}

// convert ASCII to integer
// convert integer to ASCII
int asciiToInt(string str) {
    int result = 0;
    
    for (int i = 0; i < str.length(); i ++) {
        result *= 10;
        result += (int)(str[i] -'0');
    }
    return result;
}

string intToAscii(int num) {
    string res = "";
    while (num > 0) {
        res += (char)(num%10+'0');
        num/=10;
    }
    return res;
}

// median of stream of numbers
int getMedian(int e, int &m, Heap &l, Heap &r) {
    int balance = Signum(l.GetCount(), r.GetCount());
    
    switch (balance) {
        case 1:
            // There are more on the left (max heap)
            if (e < m) { // e fits in left (max heap)
                r.Insert(l.ExtractTop());
                l.Insert(e);
            } else {
                r.Insert(e);
            }
            
            m = Average(l.GetTop(), r.GetTop());
            
            break;
            
        case 0:
            // size is the same for l and r
            if (e < m) {
                l.Insert(e);
                m = l.GetTop();
            } else {
                r.Insert(e);
                m = r.GetTop();
            }
            
            break;
            
        case -1:
            // more elements in the right (min heap)
            if (e < m) {
                l.Insert(e);
            } else {
                l.Insert(r.ExtractTop());
                r.Insert(e);
            }
            
            m = Average(l.GetTop(), r.GetTop());
            
            break;
            
        default:
            break;
    }
    
    return m;
}


// check if a path exist in a matrix
// skiing
bool pathExists(int array[5][5], int x, int y, int m, int n) {
    if (x == m || y == n || array[x][y] == 1) {
        return false;
    }
    
    if (x == m-1 && y == n-1) {
        return true;
    }
    
    return pathExists(array, x+1, y, m, n) || pathExists(array, x, y+1, m, n);
}

// unique path
// maze
// ways
int uniquePath(int x, int y, int m, int n) {
    if (x == m-1 && y == n-1)
        return 1;
    
    if (x >= m || y >= n) {
        return 0;
    }
    
    return uniquePath(x + 1, y + 1, m, n);
}

// iterative
int uniquePath_iterative(int m, int n) {
    vector<int> dp(m, 1);
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            dp[j] += dp[j-1];
        }
    }
    
    return dp[m-1];
}

// with obstacles
int uniquePathsWithObstacles(vector<vector<int>> grid) {
    
    int m = grid.size();
    int n = grid[0].size();
    
    vector<int> dp(n, 1);
    
    for (int i = 1; i < m; i++) {
        if (grid[0][m] == 1) {
            dp[i] = 0;
        } else {
            dp[i] = dp[i-1];
        }
    }
    
    for (int i = 1; i < n; i++) {
        
        dp[0] = grid[i][0] == 1 ? 0:dp[0];
        
        for (int j = 1; j < m; j++) {
            dp[j] = grid[i][j] ? 0 : dp[j-1] + dp[j];
        }
    }
    return dp[n-1];
}

// count the number of islands in an array
void turnToZero(vector<vector<int>> map, int x, int y, int m, int n) {
    if (x >= m || x < 0 || y >= n || y < 0 || map[x][y] != 1) {
        return;
    }
    map[x][y] = 0;
    turnToZero(map, x+1, y, m, n);
    turnToZero(map, x, y+1, m, n);
    turnToZero(map, x-1, y, m, n);
    turnToZero(map, x, y-1, m, n);
}

int numberOfIslands(vector<vector<int>> map, int m, int n) {
    int number = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (map[i][j] == 1) {
                number++;
                turnToZero(map, i, j, m, n);
            }
        }
    }
    
    return number;
}

// max sum
// maxSum
// maximum sub array
// max sub array
int maxSubArray(int array[], int n){
    int currMax = 0;
    int globalMax = 0;
    
    int new_starting_index = 0;
    int new_length = 0;
    
    int starting_index = 0;
    int length = 0;
    
    for (int i = 0; i < n; i++) {
        currMax += array[i];
        if (currMax < 0) {
            currMax = 0;
            new_length = 0;
            new_starting_index = i + 1;
        } else {
            new_length++;
            if (currMax > globalMax) {
                globalMax = currMax;
                length = new_length;
                starting_index = new_starting_index;
            }
        }
    }
    
    return globalMax;
}

int maxSubArray(vector<int> array) {
    int currMax = 0;
    int globalMax = 0;
    
    for (int i : array) {
        currMax = max(0, currMax + i);
        globalMax = max(globalMax, currMax);
    }
    
    return globalMax;
}

// check if two numbers are coprime to each other
// coprime is if the two number can only be commonly divided by 1
bool isCoPrime(int a, int b) {
    while (b > 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a == 1;
}

// create a better password
// replace letters
// replace char
void rePhrase(string str, int start, vector<string> &res, map<char, char>mp) {
    
    res.push_back(str);
    
    for (int i = start; i < str.size(); i++) {
        
        if (mp.find(str[i]) != mp.end()) {
            char tmp = str[i];
            str[i] = mp[str[i]];
            rePhrase(str, start+1, res, mp);
            str[i] = tmp;
        }
    }
}

vector<string> getPasswords(string password, map<char, char>mp) {
    vector<string> passwords;
    
    for (int i = 0; i < password.length(); i++) {
        char sub = mp[password[i]];
        if (sub) {
            int length = (int)passwords.size();
            for (int j = 0; j < length; j++) {
                string str = passwords[j];
                passwords.push_back(str.substr(0, i) + sub + str.substr(i+1));
            }
            passwords.push_back(password.substr(0, i) + sub + password.substr(i+1));
        }
    }
    return passwords;
}

// find the longest common substring between two texts
// O(n) solution exist in geeksforgeeks:
// http://www.geeksforgeeks.org/suffix-tree-application-5-longest-common-substring-2/
string longestCommonSubstring(string s1, string s2) {
    int start = 0;
    int length = 0;
    for (int i = 0; i < s1.length(); i++) {
        for (int j = 0; j < s2.length(); j++) {
            int x = 0;
            
            while (s1[i+x] == s2[j+x]) {
                x++;
                if ((i+x) >= s1.length() || (j+x) >= s2.length()) {
                    break;
                }
            }
            
            if (x > length) {
                length = x;
                start = i;
            }
            
        }
    }
    
    return s1.substr(start, length);
}


// dice
// given n number of dice rolls, find theprobability of rolling a 6
// rolling dice
double rollingSize(int n) {
    double probability = 1;
    
    for (int i = 0; i < n; i ++) {
        probability *= 1 - (1/6);
    }
    
    return 1 - probability;
}

// stairs
// one step or two step at a time
int combinations(int n) {
    if (n == 0) {
        return 0;
    }
    
    int fib1 = 1;
    int fib2 = 2;
    
    for (int i = 3; i <= n; i++) {
        int tmp = fib1+fib2;
        fib1 = fib2;
        fib2 = tmp;
    }
    return fib2;
}

// check if string is a rotation of another string
bool isRotation(string s1, string s2) {
    if (s1.length() != s2.length()) {
        return false;
    }
    
    s1 += s2;
    
    //    return s1.contains(s2);
    return true;
}

// print brackets
void printBrackets(string str, int l, int r) {
    if (l == 0 && r == 0) {
        cout << str << endl;
    }
    
    if (l > 0) {
        printBrackets(str+"(", l-1, r+1);
    }
    
    if (r > 0) {
        printBrackets(str+")", l, r-1);
    }
}

// find the longest substring with unique characters
string findLongestSubstring(string str) {
    bool exist[256] = {0};
    int length = 0;
    int currStart = 0;
    int start = 0;
    
    for (int i = 0; i < str.length(); i++) {
        
        if (!exist[str[i]]) {
            exist[str[i]] = true;
            
            if (i - currStart + 1 > length) {
                length = i - currStart + 1;
                start = currStart;
            }
        } else {
            while (str[currStart] != str[i]) {
                exist[str[currStart]] = false;
                currStart ++;
            }
        }
        
    }
    
    return str.substr(start, length);
}

// find the longest substring with at most two distinct characters
// 2 distinct characters
int lengthOFLongestSubstringWithTwoDistinct(string s) {
    int i = 0;
    int j = -1;
    int maxLen = 0;
    for (int k = 1; k < s.length(); k++) {
        if (s[k] == s[k-1]) {
            continue;
        }
        
        if (j>=0 && s[j] != s[k]) {
            maxLen = max(k-i, maxLen);
            i = j+1;
        }
        j = k - 1;
        
    }
    
    return max((int)s.length() - i, maxLen);
}


// random number generator
// create random 9 from random 6
//int rand9(){
//    return (rand6() - 1) * 6) + (rand6()-1))/4 + 1
//} 30 +

//int my_rand() // returns 1 to 7 with equal probability
//{
//    int i;
//    i = 5*foo() + foo() - 5;
//    if (i < 22)
//        return i%7 + 1;
//    return my_rand();
//}

//int rand9()
//{
//    int vals[5][5] = {
//        { 1, 2, 3, 4, 5 },
//        { 6, 7, 8, 9, 1 },
//        { 2, 3, 4, 5, 6 },
//        { 7, 8, 9, 0, 0 },
//        { 0, 0, 0, 0, 0 }
//    };
//    
//    int result = 0;
//    while (result == 0)
//    {
//        int i = rand5();
//        int j = rand5();
//        result= vals[i-1][j-1];
//    }
//    return result;
//}

// power function
int pow(int x, int n) {
    if (n == 0)
        return 1;
    
    if (n == 1)
        return x;
    
    int half = pow(x, n/2);
    
    if (n%2 == 0) {
        return half * half;
    } else {
        return half * half * x;
    }
}

// check if a string contains duplicate characters within k distance apart
bool containDup(string str, int k) {
    bool exist[256] = {false};
    
    for (int i = 0; i < str.length(); i++) {
        if (exist[str[i]]) {
            return true;
        }
        
        exist[str[i]] = true;
        
        if (i - k >= 0) {
            exist[str[i-k]] = false;
        }
    }
    
    return false;
}

// birthday problem
// birthday paradox
double sameBirthday(int n) {
    double np = 1;
    for (int i = 0; i < n; i++) {
        np *= (365.0 - i)/365.0;
    }
    
    return 1 - np;
}

// Returns false if no valid window is found. Else returns
// true and updates minWindowBegin and minWindowEnd with the
// starting and ending position of the minimum window.
bool minWindow(const char* S, const char *T,
               int &minWindowBegin, int &minWindowEnd) {
    int sLen = strlen(S);
    int tLen = strlen(T);
    int needToFind[256] = {0};
    
    for (int i = 0; i < tLen; i++)
        needToFind[T[i]]++;
    
    int hasFound[256] = {0};
    int minWindowLen = INT_MAX;
    int count = 0;
    for (int begin = 0, end = 0; end < sLen; end++) {
        // skip characters not in T
        if (needToFind[S[end]] == 0) continue;
        hasFound[S[end]]++;
        if (hasFound[S[end]] <= needToFind[S[end]])
            count++;
        
        // if window constraint is satisfied
        if (count == tLen) {
            // advance begin index as far right as possible,
            // stop when advancing breaks window constraint.
            while (needToFind[S[begin]] == 0 ||
                   hasFound[S[begin]] > needToFind[S[begin]]) {
                if (hasFound[S[begin]] > needToFind[S[begin]])
                    hasFound[S[begin]]--;
                begin++;
            }
            
            // update minWindow if a minimum length is met
            int windowLen = end - begin + 1;
            if (windowLen < minWindowLen) {
                minWindowBegin = begin;
                minWindowEnd = end;
                minWindowLen = windowLen;
            } // end if
        } // end if
    } // end for
    
    return (count == tLen) ? true : false;
}

// find the kth elemend of two sorted arrays
int findKthSmallest(int A[], int m, int B[], int n, int k) {
    assert(m >= 0); assert(n >= 0); assert(k > 0); assert(k <= m+n);
    
    int i = (int)((double)m / (m+n) * (k-1));
    int j = (k-1) - i;
    
    assert(i >= 0); assert(j >= 0); assert(i <= m); assert(j <= n);
    // invariant: i + j = k-1
    // Note: A[-1] = -INF and A[m] = +INF to maintain invariant
    int Ai_1 = ((i == 0) ? INT_MIN : A[i-1]);
    int Bj_1 = ((j == 0) ? INT_MIN : B[j-1]);
    int Ai   = ((i == m) ? INT_MAX : A[i]);
    int Bj   = ((j == n) ? INT_MAX : B[j]);
    
    if (Bj_1 < Ai && Ai < Bj)
        return Ai;
    else if (Ai_1 < Bj && Bj < Ai)
        return Bj;
    
    assert((Ai > Bj && Ai_1 > Bj) ||
           (Ai < Bj && Ai < Bj_1));
    
    // if none of the cases above, then it is either:
    if (Ai < Bj)
        // exclude Ai and below portion
        // exclude Bj and above portion
        return findKthSmallest(A+i+1, m-i-1, B, j, k-i-1);
    else /* Bj < Ai */
        // exclude Ai and above portion
        // exclude Bj and below portion
        return findKthSmallest(A, i, B+j+1, n-j-1, k-j-1);
}

int find_max(int freq[], bool excep[]) {
    int max_i = -1;
    int max = -1;
    for (char c = 'a'; c <= 'z'; c++) {
        if (!excep[c] && freq[c] > 0 && freq[c] > max) {
            max = freq[c];
            max_i = c;
        }
    }
    return max_i;
}


// reorder string d distance apart
void create(char* str, int d, char ans[]) {
    int n = strlen(str);
    int freq[256] = {0};
    for (int i = 0; i < n; i++)
        freq[str[i]]++;
    
    int used[256] = {0};
    for (int i = 0; i < n; i++) {
        bool excep[256] = {false};
        bool done = false;
        while (!done) {
            int j = find_max(freq, excep);
            if (j == -1) {
                cout << "Error!\n";
                return;
            }
            excep[j] = true;
            if (used[j] <= 0) {
                ans[i] = j;
                freq[j]--;
                used[j] = d;
                done = true;
            }
        }
        for (int i = 0; i < 256; i++)
            used[i]--;
    }
    ans[n] = '\0';
}

// swap string
void swap(char *a, char *b) {
    char tmp = *a;
    *a = *b;
    *b = tmp;
}

// reverse string
void reverseString(char *str, int l, int r) {
    char *start = str + l;
    char *end = str + r;
    while (start < end) {
        swap(start++, end--);
    }
}

// rotate an array
void rotate(char* str, int k) {
    int n = (int)strlen(str);
    reverseString(str, 0, n-1);
    reverseString(str, 0, k-1);
    reverseString(str, k, n-1);
}

// reverse words of a string
void reverseWords(char *str) {
    int length = strlen(str);
    reverseString(str, 0, length);
    for (int i = 0, j = 0; j <= length; j++) {
        if (j == length || str[j] == ' ') {
            reverseString(str, i, j);
            i = j + 1;
        }
    }
}

// stock
// at most k transactions
int stock_k_transaction(vector<int> prices, int k) {
    
    int global[k+1];
    int local[k+1];
    memset(global, 0, sizeof(global));
    memset(local, 0, sizeof(local));
    
    for (int i = 1; i < prices.size(); i++) {
        int difference = prices[i] - prices[i-1];
        for (int j = k; j > 0; j--) {
            local[j] = max(global[j-1] + max(difference, 0), local[j] + difference);
            global[j] = max(local[j], global[j]);
        }
    }
    
    return global[k];
}

// stock
// only one transaction
int maxProfit(vector<int> prices) {
    int diff = 0;
    int profit = 0;
    int minValue = 0;
    for (int i = 0; i < prices.size(); i++) {
        minValue = min(prices[i], minValue);
        diff = prices[i] - minValue;
        profit = max(profit, diff);
    }
    return profit;
}

// stock
// as many transaction as you like
int maxProfitInfinite(vector<int> prices) {
    int profit = 0;
    for (int i = 1; i < prices.size(); i++) {
        int diff = prices[i] - prices[i-1];
        if (diff > 0) {
            profit += diff;
        }
    }
    return profit;
}

// stock
// only two transactions
int maxProfit2k(vector<int> prices) {
    int mprof = 0;
    
    //profit before each element
    vector<int>mp;
    mp.push_back(0);
    int start = prices[0];
    for (int i = 1; i < prices.size(); i++) {
        if (mprof < prices[i] - start) {
            mprof = prices[i] - start;
        }
        
        if (prices[i] < start) {
            start = prices[i];
        }
        
        mp.push_back(mprof);
    }
    
    int end = prices[prices.size() - 1];
    for (int i = prices.size() - 2; i >= 0; i--) {
        if (mprof < end - prices[i] + mp[i]) {
            mprof = end - prices[i] + mp[i];
        }
        
        if (prices[i] > end) {
            end = prices[i];
        }
    }
    
    return mprof;
}

// Repeated DNA sequence
vector<string> findRepeatedDNASequences(string s) {
    map<int, int> m;
    vector<string> res;
    int t = 0;
    int i = 0;
    while (i < s.size()) {
        if (m[t = (t << 3 | (s[i++] & 7)) & 0x3FFFFFFF]++ == 1) {
            res.push_back(s.substr(i - 10, 10));
        }
    }
    
    return res;
}

// largest number
// rearrange the array to form the largest number
int myCompare(string X, string Y)
{
    string XY = X.append(Y);
    
    string YX = Y.append(X);
    
    return XY.compare(YX) > 0 ? 1: 0;
}

void printLargest(vector<string> arr)
{
    sort(arr.begin(), arr.end(), myCompare);
    
    for (int i=0; i < arr.size() ; i++ )
        cout << arr[i];
}

// dungeon game
// dynamic programming
// min life to survive
int minHP(vector<vector<int>> dungeon){
    int m = dungeon.size();
    int n = dungeon[0].size();
    
    int dp[m][n];
    
    //initialization
    dp[m-1][n-1] = max(0-dungeon[m-1][n-1], 0);
    
    for (int i = m-2; i >= 0; i--) {
        dp[i][n-1] = max(dungeon[i+1][n-1] - dungeon[i][n-1], 0);
    }
    
    for (int i = n-2; i >= 0; i--) {
        dp[m-1][i] = max(dungeon[m-1][i+1] - dungeon[m-1][i], 0);
    }
    
    for (int i = m-2; i >= 0; i--) {
        for (int j = n-2; j >= 0; j--) {
            dp[i][j] = max(min(dp[i][j+1], dp[i+1][j]) - dungeon[i][j], 0);
        }
    }
    
    return dp[0][0] + 1;
}

// count the factorial trailing zeros
int findTrailingZeros(int n) {
    int count = 0;
    
    for (int i = 5; n/i > 0; i *= 5) {
        count += n/i;
    }
    
    return count;
}

// excel string to number
int titleToNumber(string s) {
    int result = 0;
    for (int i = 0; i < s.length(); i++) {
        result *= 26;
        result += (s[i] - 'A' + 1);
    }
    
    return result;
}

string convertToTitle(int n) {
    if (n <=0) return "";
    string res = "";
    while (n > 0) {
        res += (char)('A' + (n-1)%26);
        n = (n-1)/26;
    }
    
    return res;
}

// two sum data structure
class twoSum {
private:
    set<int> database;
    
public:
    void add(int number){
        database.insert(number);
    }
    
    bool find(int value) {
        for (int i = value - 1; i > 0; i--) {
            if (database.find(value - i) != database.end()) {
                return true;
            }
        }
        return false;
    }
};

// two sum
// sorted
bool twoSum(vector<int> array, int target) {
    int i = 0;
    int j = array.size() - 1;
    
    while (i < j) {
        int sum = array[i] + array[j];
        if (sum == target) {
            return true;
        }
        
        if (sum > target) {
            i++;
        } else {
            j--;
        }
    }
    
    return  false;
}

// find the majority element
// find the element that happens more than half
// find the element that occurs more than half
int findCandidate(int a[], int size) {
    int maj_index = 0;
    int count = 1;
    
    for (int i = 1; i < size; i++) {
        if (a[maj_index] == a[i]) {
            count ++;
        } else {
            count --;
        }
        
        if (count == 0) {
            maj_index = i;
            count = 1;
        }
    }
    
    return a[maj_index];
}

// recurring fraction
// decimal
string fractionToDecimal(int num, int denom) {
    if (num == 0) {
        return "0";
    }
    if (denom == 0) {
        return "";
    }
    
    string solution = "";
    
    if ((num < 0) ^ (denom < 0)) {
        solution += "-";
    }
    
    long numerator = abs(num), denominator = abs(denom);
    
    long res = numerator / denom;
    solution += to_string(res);
    
    long rem = (numerator % denominator) * 10;
    if (rem == 0) {
        return solution;
    }
    
    map<long, int>mp;
    solution += ".";
    while (rem != 0) {
        if (mp.find(rem) != mp.end()) {
            int begin = mp[rem];
            string p1 = solution.substr(0, begin);
            string p2 = solution.substr(begin, solution.length());
            solution = p1 + "(" + p2 + ")";
            return solution;
        }
        
        mp.insert(pair<long, int>(rem, solution.length()));
        res = rem / denominator;
        solution += to_string(res);
        rem = (rem % denominator) * 10;
    }
    
    return solution;
}

// compare version number
// separated by a dot
// separated by .
int compareVersion(string v1, string v2) {
    
    for (int i = 0, j = 0; i < v1.size() || j < v2.size(); i++, j++) {
        int n1 = 0, n2 = 0;
        
        while (v1[i] != '.' && i < v1.size())
            n1 = n1*10 + (v1[i++] - '0');
        
        while (v2[j] != '.' && j < v2.size())
            n2 = n2 *10 + (v2[j++] - '0');
        
        if (n1 > n2) {
            return 1;
        }
        
        if (n1 < n2) {
            return -1;
        }
        
    }
    
    return 0;
}

// find missing ranges
// find missing difference
string getRange(int from, int to) {
    return (from==to) ? to_string(from) : to_string(from) + "->" + to_string(to);
}

vector<string> findMissingRanges(int vals[], int size, int start, int end) {
    vector<string> ranges;
    int prev = start - 1;
    for (int i=0; i<=size; ++i) {
        int curr = (i==size) ? end + 1 : vals[i];
        if ( curr-prev>=2 ) {
            ranges.push_back(getRange(prev+1, curr-1));
        }
        prev = curr;
    }
    return ranges;
}

// find the peak element
int findPeakElement(vector<int> num) {
    int left = 0;
    int right = num.size() - 1;
    while (left <= right) {
        if (left == right) {
            return left;
        }
        
        int mid = (left + right) / 2;
        if (num[mid] < num[mid+1]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// check if two strings are one edit distance apart
// check if two strings are one distance apart
// oneeditdistance
// onedistance
bool isOneEditDistance(string s, string t) {
    if (s.length() > t.length()) {
        swap(s, t);
    }
    if (t.length() - s.length() > 1) {
        return false;
    }
    
    bool have = false;
    
    for (int i = 0, j = 0; i < s.length(); i++, j++) {
        if (s[i] != t[j]) {
            if (have) {
                return false;
            }
            have = true;
            if (s.length() < t.length()) {
                i --;
            }
        }
    }
    
    return have || (s.length() < t.length());
}

// reverse the bits of a number
int reverseBits(int num) {
    int numOfBits = 32;
    int result = 0;
    for (int i = 0; i < numOfBits; i++) {
        if (num & (1<<i)) {
            result |= 1 << ((numOfBits - 1) - i);
        }
    }
    
    return  result;
}

// detect if two numbers have opposite signs
bool oppositeSigns(int x, int y) {
    //    return ((x ^ y) >> 31);
    return ((x ^ y) < 0);
}

// swap bits
// swap n bits at position
int swapBits(int x, int p1, int p2, int n) {
    int set1 = (x >> p1) & ((1 << n) - 1);
    int set2 = (x >> p2) & ((1 << n) - 1);
    
    int xored = (set1 ^ set2);
    xored = ((xored << p1) | (xored << p2));
    
    return x ^ xored;
}


// add two numbers using bits
int Add(int x, int y) {
    while (y!=0) {
        // carry ... & ...
        int carry = x&y;
        // value is xor
        x = x^y;
        // move carry left by one
        y = carry << 1;
    }
    
    return x;
}

// smallest of three
int smallest(int x, int y, int z) {
    int c = 0;
    while (x && y && z) {
        x--; y--; z--; c++;
    }
    return c;
}

// make an array of two elements having 0 and 1 both 0
// both zero
void changeToZero(int a[2]) {
    a[a[1]] = a[!a[1]];
}

// find the minimum in rotated binary array
// find the min in rotated binary array
int findMin(int arr[], int low, int high) {
    if (low == high) {
        return arr[low];
    }
    
    int mid = low + (high - low) / 2;
    
    if (mid < high && arr[mid + 1] < arr[mid]) {
        return arr[mid + 1];
    }
    
    // Check if mid itself is minimum element
    if (mid > low && arr[mid] < arr[mid - 1])
        return arr[mid];
    
    // Decide whether we need to go to left half or right half
    if (arr[high] > arr[mid])
        return findMin(arr, low, mid-1);
    return findMin(arr, mid+1, high);
}

//find the maximum product
//find the max product
int maxProduct(int a[], int n) {
    int res = a[0];
    int maxp = a[0];
    int minp = a[0];
    for (int i = 1; i < n; i++) {
        int tmpMax = maxp;
        int tmpMin = minp;
        maxp = max(max(tmpMax * a[i], tmpMin * a[i]), a[i]);
        minp = min(min(tmpMax * a[i], tmpMin * a[i]), a[i]);
        res = max(maxp, res);
    }
    return res;
}


// evaluate string operation
int eval(vector<string> given) {
    stack<int> operation;
    for (int i = 0; i < given.size(); i++) {
        if ((given[i][0] == '-' && given[i].size() > 1) || (given[i][0] >= '0' && given[i][0] <= '9')) {
            operation.push(atoi(given[i].c_str()));
            continue;
        }
        int op1 = operation.top();
        operation.pop();
        int op2 = operation.top();
        if(given[i] == "+") operation.push(op2+op1);
        if(given[i] == "-") operation.push(op2-op1);
        if(given[i] == "*") operation.push(op2*op1);
        if(given[i] == "/") operation.push(op2/op1);
    }
    return operation.top();
}

// recursion tree
// word break
void getSolution(string s, int start, string curr, set<string> dict, vector<string> &res) {
    if (start == s.size()) {
        res.push_back(curr.substr(0, curr.size()-1));
        return;
    }
    
    for (int i = start; i < s.size(); i++) {
        string tmp = s.substr(start, i - start + 1);
        if (dict.find(tmp) != dict.end()) {
            curr += tmp + " ";
            getSolution(s, i + 1, curr, dict, res);
            curr.resize(curr.size() - tmp.size() - 1);
        }
    }
}

vector<string> wordBreak(string s, set<string> dict) {
    vector<string> res;
    getSolution(s, 0, "", dict, res);
    return res;
}

// dynamic programming
bool wordBreakExist(string s, set<string> dict) {
    string s2 = '#' + s;
    int len = s2.size();
    vector<bool> possible(len, false);
    
    possible[0] = true;
    // i : 1->len
    for (int i = 1; i < len; i++) {
        // k : 0->i
        for (int k = 0; k < i; k++) {
            possible[i] = possible[k] && (dict.find(s2.substr(k+1, i-k))!= dict.end());
            if (possible[i]) break;
        }
    }
    
    return possible[len-1];
}

// numbers occur three times
// find one that occurs only once
// find the number that occur once
// find the number that occurs once
int singleNumber(int a[], int n) {
    int t1 = 0;
    int t2 = 0;
    int t3 = 0;
    
    for (int i = 0; i < n; i++) {
        t2 = t2 | (t1 & a[i]);
        t1 = t1 ^ a[i];
        t3 = t1&t2;
        
        // turn them three time bits into 0
        t1 = t1&~t3;
        t2 = t2&~t3;
    }
    
    return t1;
}

// candy
int getCandy(vector<int> r) {
    vector<int> lc(r.size(), 1);
    vector<int> rc(r.size(), 1);
    int res = 0;
    for (int i = 1; i < lc.size(); i++) {
        if (r[i] > r[i-1]) {
            lc[i] = lc[i-1] + 1;
        }
    }
    for (int i=rc.size()-2;i>=0;i--){
        if (r[i]>r[i+1]){
            rc[i]=rc[i+1]+1;
        }
    }
    for (int i=0;i<r.size();i++){
        res+=max(lc[i],rc[i]);
    }
    
    return res;
}

// gas station
// travel around once
// enough gas
int completeCircuit(vector<int> gas, vector<int> cost) {
    vector<int> diff(gas.size());
    
    for (int i = 0; i < gas.size(); i++) {
        diff[i] = gas[i] - cost[i];
    }
    
    int leftGas = 0, sum = 0, startNode = 0;
    for (int i = 0; i < gas.size(); i++) {
        leftGas += diff[i];
        sum += diff[i];
        if (sum < 0) {
            startNode = i + 1;
            sum = 0;
        }
    }
    
    if (leftGas < 0) {
        return  -1;
    }
    
    return startNode;
}

// palindrome
// partition string such that every substring is a palindrome
bool valid(string s, int start, int end) {
    while (start < end) {
        if (s[end--] != s[start++]) {
            return false;
        }
    }
    return true;
}

void find(string s, int start, vector<string> &r, vector<vector<string>> &res) {
    if (start == s.size()) {
        res.push_back(r);
    }
    
    for (int i = start; i < s.size(); i++) {
        string tmp = s.substr(start, i - start + 1);
        if (valid(s, start, i)) {
            r.push_back(tmp);
            find(s, i+1, r, res);
            r.pop_back();
        }
    }
}

vector<vector<string>> partition(string s) {
    vector<vector<string>> res;
    vector<string> r;
    find(s, 0, r, res);
    return res;
}

// minimum cut palindrome
// dynamic programming
int minCut(string s) {
    int len = s.size();
    int d[len + 1];
    bool p[len][len];
    
    for (int i = 0; i <= len; i++) {
        d[i] = len - i;
    }
    
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            p[i][j] = false;
        }
    }
    
    // i : len -> 0
    for (int i = len - 1; i >= 0; i--) {
        // j : i -> len
        for (int j = i; j < len; j ++) {
            if (s[i] == s[j] && (j - i < 2 || p[i+1][j-1])) {
                p[i][j] = true;
                d[i] = min(d[i], d[j+1] + 1);
            }
        }
    }
    return d[0] - 1;
}

// surrounded region
// fill the border
void fill(vector<vector<char>> board, int i, int j, char target, char c) {
    int m = board.size();
    int n = board[0].size();
    if (i < 0 || i >= m || j >= n || j < 0 || board[i][j] != target) {
        return;
    }
    
    board[i][j] = c;
    fill(board, i+1, j, target, c);
    fill(board, i, j+1, target, c);
    fill(board, i-1, j, target, c);
    fill(board, i, j-1, target, c);
}

void fill_iterative(vector<vector<char>> board, int i, int j, char target, char c) {
    int m = board.size();
    int n = board[0].size();
    if (i < 0 || i >= m || j >= n || j < 0 || board[i][j] != target) {
        return;
    }
    stack<pair<int, int>>s;
    s.push(make_pair(i, j));
    
    while (!s.empty()) {
        i = s.top().first;
        j = s.top().second;
        board[i][j] = c;
        
        s.pop();
        
        if (i>0 && board[i-1][j] == target) {
            s.push(make_pair(i-1, j));
        }
        
        if (j>0 && board[i][j - 1] == target) {
            s.push(make_pair(i, j - 1));
        }
        
        if (i < m-1 && board[i + 1][j] == target) {
            s.push(make_pair(i + 1, j));
        }
        
        if (j > n - 1 && board[i][j + 1] == target) {
            s.push(make_pair(i, j + 1));
        }
    }
}

void fillBorders(vector<vector<char>> board, char target, char c) {
    int m = board.size();
    int n = board[0].size();
    
    for (int i = 0; i < m; i++) {
        if (board[i][0] == target) {
            fill(board, i, 0, target, c);
        }
        if (board[i][n-1] == target) {
            fill(board, i, n-1, target, c);
        }
    }
    for (int j = 0; j < n; j++) {
        if (board[0][j] == target) {
            fill(board, 0, j, target, c);
        }
        if (board[n-1][j] == target) {
            fill(board, n-1, j, target, c);
        }
    }
}

void replace(vector<vector<char>> &board, char target, char c) {
    int m = board.size(), n = board[0].size();
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            if(board[i][j]==target)
                board[i][j] = c;
        }
    }
}

void solve(vector<vector<char>> &board) {
    if(board.size()<3 || board[0].size()<3) return;
    fillBorders(board, 'O', 'Y');
    replace(board, 'O', 'X');
    fillBorders(board, 'Y', 'O');
}

// longest consecutive sequence
int longestConsecutiveSequence(vector<int> num) {
    map<int, bool> mp;
    for (int i = 0; i < num.size(); i++) {
        mp[num[i]] = true;
    }
    
    int res = 0;
    for (int i = 0; i < num.size(); i++) {
        int count = 1;
        int element = num[i];
        
        mp.erase(element);
        
        while (mp.find(element + 1) != mp.end()) {
            count ++;
            element ++;
            mp.erase(element);
        }
        
        element = num[i];
        while (mp.find(element - 1) != mp.end()) {
            count ++;
            element --;
            mp.erase(element);
        }
        
        res = max(res, count);
    }
    
    return res;
}

// dynamic programming
// distinct subsequences
// rabbbit, rabbit, returns 3
int numDistinct(string s, string t) {
    int match[256];
    if (s.size() < t.size()) {
        return 0;
    }
    
    //preset
    match[0] = 1;
    for (int i = 1; i < t.size(); i++) {
        match[i] = 0;
    }
    
    // i : 1->size
    for (int i = 1; i <= s.size(); i++) {
        // j : size -> 1
        for (int j = t.size(); j >= 1; j--) {
            if (s[i-1] == t[j-1]) {
                match[j] += match[j-1];
            }
        }
    }
    
    return match[t.size()];
}


//TODO
// word ladder
class wordLadderAllSolution {
public:
    map<string,vector<string>> mp; // result map
    vector<vector<string> > res;
    vector<string> path;
    
    void findDict(string str, set<string> &dict,set<string> &next_lev){
        int sz = str.size();
        string s = str;
        for (int i=0;i<sz;i++){
            s = str;
            for (char j = 'a'; j<='z'; j++){
                s[i]=j;
                if (dict.find(s)!=dict.end()){
                    next_lev.insert(s);
                    mp[s].push_back(str);
                }
            }
        }
    }
    
    void output(string &start,string last){
        if (last==start){
            reverse(path.begin(),path.end());
            res.push_back(path);
            reverse(path.begin(),path.end());
        }else{
            cout << last << endl;
            for (int i=0;i<mp[last].size();i++){
                print_vector(mp[last]);
                path.push_back(mp[last][i]);
                output(start,mp[last][i]);
                path.pop_back();
            }
        }
    }
    
    vector<vector<string>> findLadders(string start, string end, set<string> &dict) {
        mp.clear();
        res.clear();
        path.clear();
        
        dict.insert(start);
        dict.insert(end);
        
        set<string> cur_lev;
        cur_lev.insert(start);
        set<string> next_lev;
        path.push_back(end);
        
        
        while (true){
            //delete previous level words
            for (auto it = cur_lev.begin();it!=cur_lev.end();it++){
                dict.erase(*it);
            }
            
            //find current level words that are one distance apart
            //and push it into next_lev
            for (auto it = cur_lev.begin();it!=cur_lev.end();it++){
                findDict(*it, dict, next_lev);
            }
            
            if (next_lev.empty()){
                return res;
            }
            
            // the end now exist within the next level
            if (next_lev.find(end)!=next_lev.end()){ //if find end string
                output(start,end);
                return res;
            }
            
            // move to the next level
            cur_lev.clear();
            cur_lev = next_lev;
            next_lev.clear();
            
        }
        return res;
    }
};

// word ladder
// transform s to t one letter at a time while matching dictionary
bool valid(string s, string t) {
    bool flag = false;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] != t[i]) {
            if (flag) {
                return false;
            } else {
                flag = true;
            }
        }
    }
    return true;
}

struct node {
    string str;
    int lev;
    node (string s, int l): str(s), lev(l) {}
};

int ladderLength(string start, string end, set<string> dict) {
    if (valid(start, end)) {
        return 2;
    }
    
    int level = 1;
    int rlevel = 1;
    queue<node>q;
    queue<node>rq;
    map<string, bool>mark;
    map<string, bool>rmark;
    for (auto it=dict.begin(); it!=dict.end();it++) {
        mark[*it] = false;
        rmark[*it] = false;
    }
    while (!q.empty() && !rq.empty()){
        
        if (q.size()<rq.size()){
            while (!q.empty() && q.front().lev==level){
                for (auto it=dict.begin();it!=dict.end();it++){
                    if (!mark[*it] && valid(q.front().str,*it)){
                        mark[*it]=true;
                        if (rmark[*it]){return q.front().lev+rq.back().lev;}
                        q.push(node(*it,level+1));
                    }
                }
                q.pop();
            }
            level++;
        }else{
            while (!rq.empty() && rq.front().lev==rlevel){
                for (auto it=dict.begin();it!=dict.end();it++){
                    if (!rmark[*it] && valid(*it,rq.front().str)){
                        rmark[*it]=true;
                        if (mark[*it]){return rq.front().lev+q.back().lev;}
                        rq.push(node(*it,rlevel+1));
                    }
                }
                rq.pop();
            }
            
            rlevel++;
        }
    }
    
    return 0;
}

// triangle
// find minimum path in triangle
// minimum path in triangle
// maximum path in triangle
// min path in triangle
// max path in triangle
int minPathTriangle(vector<vector<int>> tri) {
    int total[tri.size()];
    int l = tri.size() - 1;
    
    for (int i = 0; i < tri.size(); i++) {
        total[i] = tri[l][i];
    }
    
    for (int i = tri.size() - 2; i >= 0; i--) {
        for (int j = 0; j < tri[i+1].size() - 1; j++) {
            total[j] = tri[i][j] + min(total[j], total[j+1]);
        }
    }
    
    return total[0];
}

// pascal triangle
// pascal's triangle
// pascals triangle
vector<vector<int>> generate(int numRows) {
    vector<vector<int>> res;
    if (numRows == 0) return res;
    
    vector<int> r;
    r.push_back(1);
    res.push_back(r);
    if (numRows == 1) {
        return res;
    }
    
    r.push_back(1);
    res.push_back(r);
    if (numRows == 2) {
        return res;
    }
    
    for (int i = 2; i < numRows; i ++) {
        vector<int> c(i+1, 1);
        for (int j = 1; j < i; j++) {
            c[j] = res[i-1][j] + res[i-1][j-1];
        }
        res.push_back(c);
    }
    return res;
}

// pascal
// give out the kth row;
// r(k) = r(k-1) * (n+1-k)/k,
vector<int> getRow(int rowIndex) {
    vector<int> res;
    res.push_back(1);
    int n = rowIndex / 2;
    for (int i = 1; i<= n; i++) {
        double r = double(res[i-1]) * (double(rowIndex)+1-double(i))/double(i);
        res.push_back(r);
    }
    
    if (rowIndex%2 == 1) {
        int sz = res.size();
        for (int i = sz - 1; i >= 0; i--) {
            res.push_back(res[i]);
        }
    } else {
        int sz = res.size();
        for (int i=sz-2;i>=0;i--){
            res.push_back(res[i]);
        }
    }
    return res;
}

// interleaving string
// check if s1 s2 can form s3
bool isInterleave(string a, string b, string c) {
    int n = a.size();
    int m = b.size();
    
    vector<vector<bool>> A(n + 1, vector<bool>(m + 1, false));
    
    if (n + m != c.size()) return false;
    if (a.empty() && b.empty() && c.empty()) return true;
    
    for (int i = 1; i <= n; i++) {
        if (a[i-1] == c[i-1] && A[i-1][0]) {
            A[i][0] = true;
        }
    }
    
    for (int i = 1; i <= m; i++) {
        if (b[i-1] == c[i-1] && A[0][i-1]) {
            A[0][i] = true;
        }
    }
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            A[i][j] = (A[i][j-1] && (b[j-1] == c[i+j-1])) || ( A[i-1][j] && ( a[i-1] == c[i + j - 1] ));
        }
    }
    return A[n][m];
}

// valid ip address
// recurrsion tree
bool valid(string s) {
    if (s.size() == 3 && (atoi(s.c_str()) > 255 || atoi(s.c_str()) == 0)) {
        return false;
    }
    if (s.size() == 3 && s[0] == '0') {
        return false;
    }
    
    if (s.size() == 2 && atoi(s.c_str()) == 0) {
        return false;
    }
    if (s.size() == 2 && s[0] == '0') {
        return false;
    }
    
    return true;
}

void getRes(string s, string curr, vector<string> &res, int k) {
    if (k==0) {
        if (s.empty()) {
            res.push_back(curr);
        }
        return;
    }
    
    for (int i = 1; i <= 3; i++) {
        if (s.size() >= i && valid(s.substr(0, i))) {
            if (k == 1) {
                getRes(s.substr(i), curr + s.substr(0, i), res, k - 1);
            } else {
                getRes(s.substr(i), curr + s.substr(0, i) + ".", res, k - 1);
            }
        }
        
    }
    
}

// decode ways
// abcdefg
// 12345
int numDecodings(string s) {
    if (s.length() == 0) {
        return 0;
    }
    
    int nums[s.length() + 1];
    nums[0] = 1;
    nums[1] = s[0] != '0' ? 1 : 0;
    
    for (int i = 2; i <= s.length(); i++) {
        if (s[i-1] != '0') {
            nums[i] = nums[i-1];
        }
        
        int twoDigits = (s[i - 2] - '0') * 10 + s[i-1];
        if (twoDigits >= 10 && twoDigits <= 26) {
            nums[i] += nums[i - 2];
        }
    }
    
    return nums[s.length()];
}

// subsets.
// permutation
// no duplicates
void generateSub(vector<int> &s, int step, vector<vector<int>> &result,vector<int> output)
{
    if (step == s.size()) {
        result.push_back(output);
    }
    
    for(int i = step;i<s.size(); i++ )
    {
        output.push_back(s[i]);
        result.push_back(output);
        generateSub(s, i+1, result, output);
        
        output.pop_back();
        while(i<s.size()-1 && s[i] == s[i+1])
            i++;
    }
}

// original
// contain duplicates
void combination(int k, int n, int start, vector<vector<int>> &result, vector<int> curr) {
    if (start == k) {
        result.push_back(curr);
        return;
    }
    
    for (int i = start; i <= n; i++) {
        curr.push_back(i);
        combination(k, n, i+1, result, curr);
        curr.pop_back();
    }
}


vector<vector<int> > subsetsWithDup(vector<int> &S) {
    // Start typing your C/C++ solution below
    // DO NOT write int main() function
    vector<vector<int> > result;
    vector<int> output;
    if(S.size() ==0) return result;
    result.push_back(output);
    sort(S.begin(), S.end());
    generateSub(S, 0, result, output);
    return result;
}

vector<vector<int> > subsetsWithDup_iterative(vector<int> &S) {
    // Start typing your C/C++ solution below
    // DO NOT write int main() function
    sort(S.begin(),S.end());
    vector<vector<int> > res;
    vector<int> r;
    res.push_back(r);
    
    r.push_back(S[0]);
    res.push_back(r);
    
    int pre = S[0];
    int count = 1;
    
    for (int i=1;i<S.size();i++){
        int st=0;
        int sz = res.size();
        if (S[i]==pre){st = sz-count;}
        count =0;
        for (int j=st;j<sz;j++){
            r = res[j];
            r.push_back(S[i]);
            res.push_back(r);
            count++;
        }
        pre=S[i];
    }
    return res;
}

// gray code
vector<int> grayCode(int n) {
    vector<int> res;
    int size = 1<<n;
    for (int i = 0; i < size; i++) {
        res.push_back((i >> 1)^i);
    }
    
    return res;
}

// merge sorted array
void merge_sorted_array(int a[], int m, int b[], int n) {
    int k = m + n - 1;
    int i = m-1;
    int j = n-1;
    while (i >= 0 && j >= 0) {
        if (a[i] > b[j]) {
            a[k--] = a[i--];
        } else {
            a[k--] = b[j--];
        }
    }
    
    while (j>=0) {
        a[k--] = b[j--];
    }
}

// scramble string
bool isScramble(string a, string b) {
    if (a.length() != b.length()) {
        return false;
    }
    if (a.compare(b)) {
        return true;
    }
    
    // check characters;
    int chars[26] = {0};
    int len = a.length();
    for (int i = 0; i < a.length(); i++) {
        chars[a[i]] ++;
        chars[b[i]] --;
    }
    for (int i = 0; i < 26; i++) {
        if (chars[i] != 0) {
            return false;
        }
    }
    
    // more letters
    for (int i = 1; i < len; i++) {
        string a1 = a.substr(0, i);
        string a2 = a.substr(i, len);
        string b1 = b.substr(0, i);
        string b2 = b.substr(i, len);
        if (isScramble(a1, b1) && isScramble(a2, b2)) {
            return true;
        }
        
        b1 = b.substr(0, len - i);
        b2 = b.substr(len-i, len);
        if (isScramble(a1, b2) && isScramble(a2, b1)) {
            return true;
        }
    }
    
    return false;
}

// maximal rectangle
// max rectangle
// maximal retangle with ones
// max retangle with ones
void printMaxSubSquare(vector<vector<bool>> m) {
    int r = m.size();
    int c = m[0].size();
    vector<vector<bool>> s(m.size(), vector<bool>(m[0].size(), false));
    
    for (int i = 0; i<r; i++) {
        s[i][0] = m[i][0];
    }
    
    for (int i = 0; i < c; i++) {
        s[0][i] = m[0][i];
    }
    
    for (int i = 1; i < r; i++) {
        for (int j = 1; j < c; j++) {
            if (m[i][j]) {
                s[i][j] = min(min(s[i][j-1], s[i-1][j]), s[i-1][j-1]) + 1;
            } else {
                s[i][j] = 0;
            }
        }
    }
    
}

// largest rectangle in histogram
int maxArea(int a[], int n) {
    stack<int> s;
    
    int max_area = 0;
    int i = 0;
    while (i < n) {
        if (s.empty() || a[s.top()] < a[i])
            s.push(i++);
        else {
            int index = s.top();
            s.pop();
            
            // empty -> i
            // not empty -> math
            int curr_area = a[index] * (s.empty() ? i : i - s.top() - 1);
            
            max_area = max(max_area, curr_area);
        }
    }
    
    while (!s.empty()) {
        int index = s.top();
        s.pop();
        
        int curr_area = a[index] * (s.empty() ? i : i - s.top() - 1);
        max_area = max(max_area, curr_area);
    }
    return max_area;
}

// search in rotated array
// rotated binary search
int rotatedBinarySearch(int a[], int n, int element) {
    int l = 0;
    int r = n-1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if ( a[m] == element ) {
            return m;
        }
        
        if ( a[l] < a[m] ) {
            if (a[l] < element && element < a[m]) {
                r = m-1;
            } else {
                l = m + 1;
            }
        } else {
            if (a[m] < element && element < a[r]) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
    }
    return -1;
}

// find minimum in rotated array
// find min in rotated array
//int findMin(int arr[], int low, int high) {
//    if (high == low) return arr[low];
//    
//    int mid = low + (high - low)/2;
//    
//    if (mid < high && arr[mid + 1] < arr[mid]) return arr[mid+1];
//    
//    if (mid > low && arr[mid-1] > arr[mid]) return arr[mid];
//    
//    if (arr[high] > arr[mid]) return findMin(arr, low, mid-1);
//        
//    return findMin(arr, mid+1, low);
//}

// remove duplicate from array
// keep at most two
int removeDup(int a[], int n) {
    if (n < 3) {
        return n;
    }
    
    int end = 1; // *
    for (int i = 2; i < n; i++) { // *
        if (a[i] != a[end-1]) {
            a[++end] = a[i];
        }
    }
    return end + 1;
}

// remove duplicate
// all
int removeDuplicates(int a[], int n) {
    int end = 0;
    for (int i = 1; i < n; i++) {
        if (a[i] != a[end]) {
            a[++end] = a[i];
        }
    }
    return end + 1;
}

// word search in matrix
// word search in 2d array
bool search(vector<vector<char>> board, int i, int j, string str, vector<vector<bool>> mask) {
    
    if (str.size() == 0) {
        return true;
    } else {
        if (i > 0 && board[i - 1][j] == str[0] && mask[i - 1][j] == 0) {
            mask[i - 1][j] = true;
            if (search(board, i-1, j, str.substr(1), mask))
                return true;
        }
        if ((i<board.size()-1)&&(board[i+1][j]==str[0])&&(mask[i+1][j]==false)){
            mask[i+1][j]=true;
            if (search(board,i+1,j,str.substr(1),mask)){
                return true;
            }
            mask[i+1][j]=false;
        }
        if ((j>0)&&(board[i][j-1]==str[0])&&(mask[i][j-1]==false)){
            mask[i][j-1]=true;
            if (search(board,i,j-1,str.substr(1),mask)){
                return true;
            }
            mask[i][j-1]=false;
        }
        if ((j<board[0].size()-1)&&(board[i][j+1]==str[0])&&(mask[i][j+1]==false)){
            mask[i][j+1]=true;
            if (search(board,i,j+1,str.substr(1),mask)){
                return true;
            }
            mask[i][j+1]=false;
        }
    }
    return false;
}

bool exist(vector<vector<char>> board, string word) {
    for (int i = 0; i < board.size(); i++) {
        for (int j = 0; j < board[0].size(); j++) {
            if (word[0] == board[i][j]) {
                if (word.size() == 1) {
                    return true;
                } else {
                    vector<vector<bool> > mask(board.size(),vector<bool>(board[0].size(),false));
                    mask[i][j] = true;
                    if (search(board, i, j, word.substr(1), mask)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool searchMatrix(vector<vector<int>> matrix, int target) {
    int l = 0;
    int r = matrix.size() - 1;
    
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (matrix[mid][0] == target) {
            return true;
        }
        if (matrix[mid][0] > target) {
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    
    int row = r;
    l = 0;
    r = matrix[0].size() - 1;
    while(l<=r)
    {
        int mid = (l+r)/2;
        if(matrix[row][mid] == target) return true;
        if(matrix[row][mid] > target)
        {
            r = mid-1;
        }
        else
        {
            l = mid+1;
        }
    }
    return false;
}

// simpify path
string simplifyPath(string path) {
    int i = 0;
    while (i < path.size() - 1) {
        if (path[i] == '/' && path[i+1] == '/') {
            path.erase(i, 1);
        } else {
            i++;
        }
    }
    
    if (path[path.size() - 1] != '/') {
        path += '/';
    }
    
    stack<string> dirs;
    string str = "";
    int flag = 0;
    for (int i = 0; i < path.size(); i ++) {
        if (path[i] == '/') {
            flag ++;
        }
        if (flag == 1)
            str += path[i];
        if (flag == 2) {
            if (str=="/.." && !dirs.empty()){
                dirs.pop();
            }
            if (str!="/." && str!="/.."){
                dirs.push(str);
            }
            flag=1;
            str="/";
        }
    }
    
    if (dirs.empty()){return "/";}
    str="";
    while (!dirs.empty()){
        str=dirs.top()+str;
        dirs.pop();
    }
    return str;
}

// regex
// valid number
// check if number is valid
// validnumber
bool isNumber(string s) {
    // [-+]?(\d+\.?|\.\d+)\d*(e[-+]?\d+)?
    
    string rStr = "[-+]?(\\d+\\.?|\\.\\d+)\\d*(e[-+]?\\d+)?";
    
    return regex_match(s.begin(), s.end(), regex(rStr));
}

// add numbers
// add binarys
// add strings
string addStrings(string a, string b) {
    int carry = 0;
    string res;
    for (int i = a.size() - 1, j = b.size() - 1; i >= 0 || j>=0; i--, j--) {
        int ai = i>=0 ? a[i] - '0' : 0;
        int bi = j>=0 ? b[i] - '0' : 0;
        int val = (ai + bi + carry)%10;
        carry = (ai + bi + carry)/10;
        res.insert(res.begin(), val + '0');
    }
    if (carry) {
        res.insert(res.begin(), carry + '0');
    }
    
    return res;
}

// minimum path sum
// maximum path sum
// min path sum
// max path sum
// dynamic programming
int minPathSum(vector<vector<int>> grid) {
    int row = grid.size();
    int col = grid[0].size();
    vector<int> res(col, INT_MAX);
    
    res[0] = 0;
    
    for (int i = 0; i<row; i++) {
        res[0] += grid[i][0];
        for (int j = 1; j < col; j++) {
            res[j] = min(res[j-1], res[j]) + grid[i][j];
        }
    }
    return res[col-1];
}

// create spiral matrix
vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> res(n, vector<int>(n, 0));
    
    int x = 0;
    int y = 0;
    int i = 1;
    res[0][0] = i++;
    while (i <= n*n) {
        while (x + 1 < n && res[x + 1][y] == 0) {
            res[++x][y] = i++;
        }
        while (y+1<n && res[x][y + 1]==0){   // keep going down
            res[x][++y]=i++;
        }
        while (x-1>=0 && res[x-1][y]==0){  // keep going up
            res[--x][y]=i++;
        }
        while (y-1>=0 && res[x][y-1]==0){  // keep going left
            res[x][--y]=i++;
        }
    }
    return res;
}

// length of last word
int lengthOfLastWord(string s) {
    int res = 0;
    for (int i = s.size() - 1; i >= 0; i--) {
        if (s[i] == ' ') {
            return res;
        } else {
            res++;
        }
    }
    
    return res;
}

// insert interval
// insert a new element
struct Interval {
    int start;
    int end;
    Interval() : start(0), end(0) {}
    Interval(int s, int e) : start(s), end(e) {}
};

vector<Interval> insertInterval(vector<Interval> intervals, Interval newInterval) {
    vector<Interval> res;
    vector<Interval> :: iterator it;
    for (it = intervals.begin(); it != intervals.end(); it++) {
        if (newInterval.start < it->start) {
            intervals.insert(it, newInterval);
            break;
        }
    }
    
    if (it == intervals.end()) intervals.insert(it, newInterval);
    res.push_back(*intervals.begin());
    
    for (it = intervals.begin() + 1; it != intervals.end(); it++) {
        if (res.back().end >= it->start) {
            res.back().end = max(res.back().end, it->end);
        } else {
            res.push_back(*it);
        }
    }
    return res;
}

// merge intervals
bool myfunc(const Interval &a, const Interval &b){
    return (a.start < b.start);
}

vector<Interval> mergeInterval(vector<Interval> intervals) {
    vector<Interval> res;
    
    sort(intervals.begin(), intervals.end(), myfunc);
    
    res.push_back(intervals[0]);
    for (int i = 1; i < intervals.size(); i++) {
        if (res.back().end >= intervals[i].start) {
            res.back().end = max(res.back().end, intervals[i].end);
        } else {
            res.push_back(intervals[i]);
        }
    }
    
    return res;
}

// jump game
// can jump
bool canJump(int a[], int n) {
    if (n == 0 || n == 1) return true;
    
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (i <= m) {
            m = max(m, a[i] + i);
        }
        if (m >= n-1)
            return true;
    }
    return false;
}

// jump steps
int jump(int a[], int n) {
    if (n == 0 || n == 1)
        return 0;
    
    int m = 0;
    int i = 0;
    int njump = 0;
    
    while (i < n) {
        m = max(m, a[i] + i);
        if (m > 0) {
            njump ++;
        }
        if (m >= n-1) {
            return njump;
        }
        
        int tmp = 0;
        for (int j = i + 1; j <= m; j++) {
            if (j + a[j] > tmp) {
                tmp = a[j] + j;
                i = j;
            }
        }
    }
    return njump;
}

vector<int> spiralOrder(vector<vector<int> > &matrix) {
    // Start typing your C/C++ solution below
    // DO NOT write int main() function
    vector<int> res;
    if (matrix.empty()){return res;}
    if (matrix.size()==1){return matrix[0];}
    int m = matrix.size();
    int n = matrix[0].size();
    vector<vector<bool> > mask(m,vector<bool>(n,false));
    int i=0;
    int j=0;
    int k=0;
    res.push_back(matrix[i][j]);
    mask[0][0]=true;
    while (k<m*n-1){
        while ((j+1<n)&&(mask[i][j+1]==false)){
            j++;
            k++;
            res.push_back(matrix[i][j]);
            mask[i][j]=true;
        }
        
        while ((i+1<m)&&(mask[i+1][j]==false)){
            i++;
            k++;
            res.push_back(matrix[i][j]);
            mask[i][j]=true;
        }
        
        while ((j>0)&&(mask[i][j-1]==false)){
            j--;
            k++;
            res.push_back(matrix[i][j]);
            mask[i][j]=true;
        }
        
        while ((i>0)&&(mask[i-1][j]==false)){
            i--;
            k++;
            res.push_back(matrix[i][j]);
            mask[i][j]=true;
        }
    }
    return res;
}

// n-queens
// n queens
// queens
bool isValid(int a[], int r) {
    for (int i = 0; i < r; i++) {
        if (a[i] == a[r] || abs(a[i] - a[r]) == abs(r - i)) {
            return false;
        }
    }
    return true;
}

void printRes(int a[], int n) {
    vector<string> r;
    for (int i = 0; i < n; i++) {
        string str(n, '.');
        str[a[i]] = 'Q';
        r.push_back(str);
    }
    // res.push_back(r);
}

void nqueens(int a[], int curr, int n, int &res) {
    if (curr == n) {
        // printRes(a, n);
        res++;
        return;
    }
    
    for (int i = 0; i < n; i++) {
        a[curr] = i;
        if (isValid(a, curr)) {
            nqueens(a, curr + 1, n, res);
        }
    }
}

int totalQueens(int n) {
    int res = 0;
    int a[n];
    nqueens(a, 0, n, res);
    return res;
}


// print all anagrams together.
struct anagram {
    string s;
    int index;
};

bool compareAnagram(anagram a, anagram b) {
    return (a.s.compare(b.s) >= 1);
}

void printAnagram(vector<string> arr) {
    int n = arr.size();
    
    vector<anagram> p;
    
    for (int i = 0; i < n; i++) {
        anagram word;
        word.s = arr[i];
        word.index = i;
        p.push_back(word);
        std::sort(word.s.begin(), word.s.end());
    }
    
    std::sort(p.begin(), p.end(), compareAnagram);
    for (int i = 0; i < n; i++)
        cout << arr[p[i].index] << endl;
}

// rotate an image
// rotate matrix
// flip diagonally
// flip vertically
void rotate(vector<vector<int>> matrix) {
    int len = matrix[0].size();
    
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - i; j++) {
            swap(matrix[i][j], matrix[len - 1 - j][len - 1 - i]);
        }
    }
    
    for (int i = 0; i < len / 2; i++) {
        for (int j = 0; j < len; j++) {
            swap(matrix[i][j], matrix[len - i - 1][j]);
        }
    }
}


// multiply divisor(f) by 2
// multiply c = 1 by 2
//
// while dividend(f) greater than divisor
// // while dividend(f) > divisor(f) , dividend(f) -= divisor(f), res += c
// reduce divisor(f), c by 2
int divide(int dividend, int divisor) {
    // Start typing your C/C++ solution below
    // DO NOT write int main() function
    int sign = 1;
    if (dividend<0){sign = -sign;}
    if (divisor<0){sign = -sign;}
    
    unsigned long long tmp = abs((long long)dividend);
    unsigned long long tmp2 = abs((long long)divisor);
    
    unsigned long c = 1;
    while (tmp>tmp2){
        tmp2 <<= 1;
        c <<= 1;
    }
    
    int res = 0;
    while (tmp>=abs((long long)divisor)){
        while (tmp>=tmp2){
            tmp -= tmp2;
            res += c;
        }
        tmp2 >>= 1;
        c >>= 1;
    }
    
    return sign*res;
}

int main() {
    divide(15, 3);
    return 0;
}

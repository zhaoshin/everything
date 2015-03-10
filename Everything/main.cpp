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

#include "medianStream.h"

using namespace std;

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
        letter_count[str2[i]]++;
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
        subsets(str, i+1, curr+str[i]);
    }
}

// TODO
// print all subsets
// iterative
vector<vector<char>> subsets(vector<char> set) {
    vector<vector<char>> subsets;
    subsets.push_back(vector<char>());
    for (char o : set) {
        vector<vector<char>>tmp;
        
        for (vector<char> s : subsets) {
            tmp.push_back(s);
        }
        
        for (vector<char> s : tmp) {
            s.push_back(o);
        }
        subsets.insert(subsets.end(), tmp.begin(), tmp.end());
    }
    
    return subsets;
}

// get all permutation of a string
void permutation(string arr, int curr, int size, set<string> &set)
{
    if(curr == size-1) {
        if (set.find(arr) == set.end()) {
            set.insert(arr);
        }
    } else {
        for(int i=curr; i<size; i++)
        {
            swap(arr[curr], arr[i]);
            permutation(arr, curr+1, size, set);
            swap(arr[curr], arr[i]);
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


// random number generator
// create random 9 from random 6
//int rand9(){
//    return ((())rand6() - 1) * 6) + (rand6()-1))/4 + 1
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
            dp[i][j] = max(min(dp[i][j+1], dp[i+1][j]), 0);
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

// find the maximum in a sliding window of size
void maxSlidingWindow(int A[], int n, int w, int B[]) {
    deque<int> Q;
    for (int i = 0; i < w; i++) {
        while (!Q.empty() && A[i] >= A[Q.back()])
            Q.pop_back();
        Q.push_back(i);
    }
    for (int i = w; i < n; i++) {
        B[i-w] = A[Q.front()];
        while (!Q.empty() && A[i] >= A[Q.back()])
            Q.pop_back();
        while (!Q.empty() && Q.front() <= i-w)
            Q.pop_front();
        Q.push_back(i);
    }
    B[n-w] = A[Q.front()];
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

int main(int argc, const char * argv[]) {

    return 0;
}

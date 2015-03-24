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

void subSetSum(int target, int sum, int candidates[], int start, int sz, int index[], int n) {
    if (sum>target) {
        return;
    }
    if (sum == target) {
        // operate
    }
    for (int i = start; i < sz; i++) {
        index[n] = candidates[i];
        subSetSum(target, sum + candidates[i], candidates, i, sz, index, n+1);
    }
}

// recursion tree
void subSetSum (int target, int candidates[], int size) {
    int index[1000];
    subSetSum(target, 0, candidates, 0, size, index, 0);
}

// remove duplicate in string
// remove duplicate string
void removeDuplicate(char *str) {
    bool exist[256] = {false};
    
    exist[str[0]] = true;
    int tail = 1;
    
    for (int i = 1; i < strlen(str); i++) {
        if (!exist[str[i]]) {
            str[tail++] = str[i];
            exist[str[i]] = true;
        }
    }
    str[tail] = '\0';
}

// prime sieve
void markPrime(vector<bool> arr, int a) {
    int i = 2;
    int num;
    while ((num = i * a) <= arr.size()) {
        arr[num] = true;
        i++;
    }
}

void primeSeive(int n) {
    if (n < 2) {
        return;
    }
    vector<bool> arr(n+1, false);
    
    for (int i = 2; i <= n; i++) {
        if (!arr[i]) {
            cout << i << endl;
            markPrime(arr, i);
        }
    }
}

// array mutiplication
void array_multi(int a[], int output[], int n) {
    int left = 1;
    int right = 1;
    for (int i = 0; i < n; i++) {
        output[i] = 1;
    }
    
    for (int i = 0; i < n; i++) {
        output[i] *= left;
        output[n - i - 1] += right;
        left *= a[i];
        right *= a[n - i - 1];
    }
}

// bubble sort
void bubbleSort(int *array, int length) {
    for (int i = 1; i < length; i++) {
        for (int j =0; i < length - 1; j++) {
            if (array[j + 1] < array[j]) {
                swap(array[j], array[j+1]);
            }
        }
    }
}

// coin change
int coinChange(vector<int> coins, int change) {
    vector<int> res(change + 1, INT_MAX);
    vector<int> seq(change + 1);
    res[0] = 0;
    
    for (int i = 0; i <= change; i++) {
        for (int j = 0; j < coins.size(); j++) {
            if (i <= coins[j] && (1 + res[i - coins[j]] < res[i])) {
                res[i] = 1 + res[i - coins[j]];
                seq[i] = j;
            }
        }
    }
    
    cout << "Coins: ";
    int j = change;
    while (j > 0) {
        cout << coins[seq[j]];
        j -= coins[seq[j]];
    }
    
    return res[change];
}

// intersect
// overlap
vector<int> overlap(int a[], int m, int b[], int n) {
    vector<int> res;
    int i, j;
    while (i < m && j < n) {
        if (a[i] < b[j])
            i++;
        else if (a[i] > b[j])
            j++;
        else {
            res.push_back(a[i]);
            i++;
            j++;
        }
    }
    return res;
}

// get max
int getMax(int a, int b) {
    int diff = a - b;
    int k = (diff >> 31) & 1;
    return b + k * diff;
}

// same count
unsigned nexthi_same_count_ones(unsigned a) {
    /* works for any word length */
    unsigned c = (a & -a);
    unsigned r = a+c;
    return (((r ^ a) >> 2) / c) | r;
}

// longest palindrome
string preProcess(string s) {
    string res = "^";
    for (int i = 0; i < s.length(); i++) {
        res += '#' + s[i];
    }
    res += "#$";
    return res;
}

string longestPalindrome(string s) {
    string T = preProcess(s);
    int n = (int) T.length();
    int *P = new int[n];
    int C = 0, R = 0;
    for (int i = 1; i < n-1; i++) {
        int i_mirror = 2*C-i; // equals to i' = C - (i-C)
        
        P[i] = (R > i) ? min(R-i, P[i_mirror]) : 0;
        
        // Attempt to expand palindrome centered at i
        while (T[i + 1 + P[i]] == T[i - 1 - P[i]])
            P[i]++;
        
        // If palindrome centered at i expand past R,
        // adjust center based on expanded palindrome.
        if (i + P[i] > R) {
            C = i;
            R = i + P[i];
        }
    }
    
    // Find the maximum element in P.
    int maxLen = 0;
    int centerIndex = 0;
    for (int i = 1; i < n-1; i++) {
        if (P[i] > maxLen) {
            maxLen = P[i];
            centerIndex = i;
        }
    }
    delete[] P;
    
    return s.substr((centerIndex - 1 - maxLen)/2, maxLen);
}

bool isPalindrome(int x) {
    int power = 1;
    while (x / power *10 > 0) {
        power*=10;
    }
    
    while (x) {
        int l = x / power;
        int r = x%10;
        if (l != r) return false;
        
        x = x%power / 10;
        power /= 100;
    }
    
    return true;
}

// prime factors
void primeFactors(int n) {
    while (n%2 == 0) {
        cout << 2 << endl;
        n /= 2;
    }
    
    for (int i = 3; i <= sqrt(n); i += 2) {
        while (n%i == 0) {
            cout << i << endl;;
            n = n/i;
        }
    }
    if (n>2) {
        cout << 2 << endl;
    }
}

// volume of water
int volume(int a[], int n) {
    int volume = 0;
    int leftMax = a[0];
    int rightMax = a[n-1];
    
    int left = 0;
    int right = n-1;
    while (left < right) {
        leftMax = max(leftMax, a[left]);
        
        rightMax = max(rightMax, a[right]);
        
        if (leftMax < rightMax) {
            volume += leftMax;
            left ++;
        } else {
            volume += rightMax;
            right--;
        }
    }
    return volume;
}

// find kth smallest in two sorted array
// assume this inequality:
// i + j + 1 = k;
int kthSmallest(int a[], int m, int b[], int n, int k) {
    int i = m/2;
    int j = k - 1 - i;
    
    int Ai_1 = (i<=0) ? INT_MIN : a[i-1];
    int Bj_1 = (j<=0) ? INT_MIN : b[j-1];
    int Ai = (i >= m) ? INT_MAX : a[i];
    int Bj = (j >= m) ? INT_MAX : b[j];
    
    if (Bj_1 < Ai && Ai < Bj) {
        return Ai;
    } else if (Ai_1 < Bj && Bj < Ai) {
        return Bj;
    }
    
    if (Ai < Bj) {
        return kthSmallest(a + i + 1, m - i - 1, b, n, k - i - 1);
    } else {
        return kthSmallest(a, m, b + j + 1, n - i - 1, k - j - 1);
    }
}

void maxSlidingWindow(int a[], int n, int w, int b[]) {
    deque<int> q;
    for (int i = 0; i < w; i++) {
        while (!q.empty() && a[i] >= a[q.back()]) {
            q.pop_back();
        }
        q.push_back(i);
    }
    
    for (int i = w; i < n; i++) {
        b[i - w] = a[q.front()];
        while (!q.empty() && a[i] >= a[q.back()]) {
            q.pop_back();
        }
        q.push_back(i);
        
        //update size of q
        while (!q.empty() && q.front() <= i-w) {
            q.pop_front();
        }
    }
    b[n - w] = a[q.front()];
}

// number of double square elements
int doubleSquare(int num) {
    int total = 0;
    int upper = sqrt(num);
    
    for (int i = 0; i <= upper; i++) {
        double j = sqrt(double(num) - i * i);
        if (j - (int)j == 0) {
            total ++ ;
        }
    }
    
    return total;
}

// search in array for x, limit at size n
int search(int arr[], int x, int n)
{
    for (int i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return -1;
}

// Prints postorder traversal from given inorder and preorder traversals
void printPostOrder(int in[], int pre[], int n)
{
    // The first element in pre[] is always root, search it
    // in in[] to find left and right subtrees
    int root = search(in, pre[0], n);
    
    // If left subtree is not empty, print left subtree
    if (root != 0)
        printPostOrder(in, pre+1, root);
    
    // If right subtree is not empty, print right subtree
    if (root != n-1)
        printPostOrder(in+root+1, pre+root+1, n-root-1);
    
    // Print root
    cout << pre[0] << " ";
}

void printPreOrder(int in[], int post[], int n)
{
    // The first element in pre[] is always root, search it
    // in in[] to find left and right subtrees
    int root = search(in, post[n-1], n);
    
    // Print root
    cout << post[n-1] << " ";
    
    // If left subtree is not empty, print left subtree
    if (root != 0)
        printPreOrder(in, post, root);
    
    // If right subtree is not empty, print right subtree
    if (root != n-1)
        printPreOrder(in+root+1, post+root, n-root-1);
}

bool isMatch(char *str, const char* pattern) {
    while (*pattern)
        if (*str++ != *pattern++)
            return false;
    return true;
}

// replace with x


void replace(char str[], const char *pattern) {
    if (str == NULL || pattern == NULL) return;
    char *pSlow = str, *pFast = str;
    int pLen = strlen(pattern);
    while (*pFast != '\0') {
        bool matched = false;
        while (isMatch(pFast, pattern)) {
            matched = true;
            pFast += pLen;
        }
        if (matched)
            *pSlow++ = 'X';
        // tricky case to handle here:
        // pFast might be pointing to '\0',
        // and you don't want to increment past it
        if (*pFast != '\0')
            *pSlow++ = *pFast++;  // *p++ = (*p)++
    }
    // don't forget to add a null character at the end!
    *pSlow = '\0';
}

// fibernacci
long long fib1(int n) {
    long long fib1 = 1;
    long long fib2 = 1;
    long long fibn;
    for (int i = 2; i <= n; i++) {
        fibn = fib1 + fib2;
        fib1 = fib2;
        fib2 = fibn;
    }
    
    return fib2;
}

// F(2k) = a(2b - a)
// F(2k+1) = a^2 + b^2
long fib_recursive(int n) {
    if (n<=2) {
        return 1;
    }
    
    double half = n/2;
    
    long a = fib_recursive(half);
    long b = fib_recursive(half + 1);
    
    if (n%2) {
        return a*a + b*b;
    } else {
        return a*(2*b - a);
    }
}

long fib_iterative(int n){
    if (n<=2) {
        return 1;
    }
    
    int h = n/2, mask = 1;
    
    while (mask <= h) {
        mask <<=1;
    }
    
    mask >>= 1;
    long long a = 1, b = 1, c;
    
    while(mask)
    {
        c = a*a+b*b;        // F(2k+1)
        if (n&mask)
        {
            b = b*(b+2*a);  // F(2k+2)
            a = c;          // F(2k+1)
        } else {
            a = a*(2*b-a);  // F(2k)
            b = c;          // F(2k+1)
        }
        mask >>= 1;
    }
    return a;
}

// bit map
// bit set
// bitmap
// bitset
void set_bit(char *b, int i) {
    b[i/8] |= 1 << (i & 7);
}

void unset_bit(char *b, int i) {
    b[i/8] &= ~(1 << (i & 7));
}

int get_bit(char *b, int i) {
    return b[i/8] & (1<<(i&7));
}

// regex match pattern
bool match_regex(string input, string pattern) {
    vector<int> counts(26, 0);
    string regexStr;
    int uniqueCount = 0;
    for (auto c : pattern) {
        if (counts[c - 'a'] == 0) {
            regexStr += "(.+)";
            counts[c -'a'] = ++uniqueCount;
        } else {
            regexStr += "\\" + to_string( counts[c - 'a'] );
        }
    }
    
    return regex_match(input.begin(), input.end(), regex(regexStr));
}


// minimum adjust so the elements are target distane apart
#define maxTarget 100
int minAdjustmentCost(vector<int> a, int target) {
    
    int cur = 0;
    int dp[2][maxTarget + 1];
    memset(dp, 0, sizeof(dp));
    
    for (int i = 0; i < a.size(); i++) {
        int next = cur^1;
        for (int j = 1; j <=maxTarget; j++) {
            dp[next][j] = INT_MAX;
            for (int k = max(j-target, 1); k <= min(j+target, maxTarget); k++) {
                dp[next][j] = min( dp[next][j], dp[cur][k] + abs(a[i] - j) );
            }
        }
        cur ^= 1;
    }
    
    int res = INT_MAX;
    for (int i = 1; i <= maxTarget; i++) {
        res = min(res, dp[cur][i]);
    }
    
    return res;
}

// find cubes
// find three numbers that cube into a target
// find three numbers that cube into n
void findCubs(int n) {
    int i, j, k, i3, j3, k3;
    int max_cube = pow((double)n, (double)(1/3));
    
    for (int i = 1; i <= max_cube; i++) {
        i3 = i * i * i;
        for (j = i+1; j<=max_cube; j++) {
            j3 = j*j*j;
            for (k = i+1; k<=max_cube; k++) {
                if (k != j ) {
                    
                    k3 = k*k*k;
                    double number = i3 + j3 - k3;
                    double cube_root = pow((double)(number), (double)(1/3));
                    
                    if (pow(cube_root, 3) == number && cube_root - (int)cube_root == 0 ) {
                        printf("%d = %d^3 + %d^3 = %d^3 + %d^3\n", i3+j3 , i,j,k,(int)cube_root );
                    }
                    
                }
            }
        }
    }
}

int getAbs(int n) {
    int mask = n >> 31;
    return ((n^mask) - mask);
}

// rotate number
int leftRotate(int n, int d) {
    return (n << d| n >> (32 - d));
}

int count(int n, int d) {
    int res = 0;
    while (n) {
        if (n%10 == d) {
            res ++;
        }
        n /= 10;
    }
    return res;
}

pair<int, int> bookPage(int d, int k) {
    pair<int, int>res;
    int i = 1, cnt=0;
    while (cnt < k) {
        cnt += count(i, d);
        i++;
    }
    
    res.first = i==1 ? 1:i-1;
    while (count(i, d) == 0) {
        i++;
    }
    
    res.second = i-1;
    return res;
}

// kmp
void computerLPSArray(char *pat, int M, int *lps) {
    int len = 0; // length of longest prefix suffix
    int i;
    
    lps[0] = 0;
    i = 1;
    while (i < M) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len!=0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void kmpSearch(char *pat, char *txt) {
    int M = strlen(pat);
    int N = strlen(txt);
    
    int lps[M];
    int j = 0;
    
    computerLPSArray(pat, M, lps);
    
    int i = 0;
    
    while (i<N) {
        if (pat[j] == txt[i]) {
            j++;
            i++;
        }
        if (j == M) {
            printf("Found pattern at %d", i-j);
            j = lps[j-1];
        } else if (i < N && pat[j] != txt[i]) {
            if (j !=0 ) {
                j = lps[j-1];
            } else
                i++;
        }
    }
}

// minimum edit distance
// min edit
int minDistance(string a, string b) {
    int m = a.size();
    int n = b.size();
    
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    
    for (int i = 0; i <= m; i++) {
        dp[i][0] = i;
    }
    
    for (int j = 0; j <= n; j++) {
        dp[0][j] = j;
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (a[i-1] == b[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = min(min(dp[i][j-1], dp[i-1][j]), dp[i-1][j-1]) + 1;
            }
        }
    }
    return dp[m][n];
}

// get the kth permutation
// get the nth permutation
string getPermutation(int n, int k) {
    vector<int> nums(n);
    int permCount =1;
    for(int i =0; i< n; i++)
    {
        nums[i] = i+1;
        permCount *= (i+1);
    }
    // change K from (1,n) to (0, n-1) to accord to index
    k--;
    string targetNum;
    
    for(int i =0; i< n; i++)
    {
        permCount = permCount/ (n-i);
        int choosed = k / permCount;
        targetNum.push_back(nums[choosed] + '0');
        //restruct nums since one num has been picked
        for(int j =choosed; j< n-i; j++)
        {
            nums[j]=nums[j+1];
        }
        k = k%permCount;
    }
    return targetNum;
}

//http://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/
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
#include <stdio.h>
#include <string>

using namespace std;

// wild car matching
// ? match any single
// * match empty to any sequence
bool isMatch(char *s, char *p) {
    char *star = NULL;
    char *ss = s;
    while (*s) {
        if (*p == '?' || *p == *s) {
            s++;
            p++;
            continue;
        }
        
        if (*p == '*') {
            star = p++;
            ss = s;
            continue;
        }
        
        if (star) {
            p = star + 1;
            s = ++ss;
            continue;
        }
        
        return false;
    }
    
    while (*p == '*') {
        p++;
    }
    
    return !*p;
}

// first missing positive
int findFirstMissingPositive(int a[], int n) {
    for (int i = 0; i < n; i++) {
        while (a[i] != i+1) {
            if (a[i] <= 0 || a[i] > n || a[i] == a[a[i] - 1]) {
                break;
            }
            // swap a[i] with a[a[i]-1]
            int tmp = a[i];
            a[i] = a[tmp - 1];
            a[tmp - 1] = tmp;
        }
    }
    
    for (int i = 0; i < n; i++) {
        if (a[i] != i+1) {
            return i+1;
        }
    }
    
    return n+1;
}

// trap rain water
// trap water
// container water
// contain water
int trap(int A[], int n) {
    int l[n];
    int r[n];
    
    int water =0;
    
    l[0]=0;
    for (int i=1;i<n;i++){
        l[i]= max(l[i-1], A[i-1]);
    }
    
    r[n-1] = 0;
    for (int i=n-2;i>=0;i--){
        r[i]=max(r[i+1],A[i+1]);
    }
    
    for (int i=0;i<n;i++){
        if (min(l[i],r[i])-A[i] >0 ){
            water += min(l[i],r[i])-A[i];
        }
    }
    
    return water;
}

// count and say
string cas(string str) {
    string res;
    
    char ch = str[0];
    int count = 1;
    for (int i = 1; i < str.size(); i++) {
        if (str[i] == ch) {
            count++;
        } else {
            char countCh = count + '0';
            res += countCh + ch;
            
            ch = str[i];
            count = 1;
        }
    }
    return res;
}

string countAndSay(int n) {
    if (n == 1) {
        return "1";
    }
    
    string res = "1";
    for (int i = 1; i < n; i++) {
        res = cas(res);
    }
    return res;
}

// search and insert
// search insert location
// search insertion location
int searchInsert(int A[], int n, int target) {
    int l = 0;
    int r = n-1;
    
    while (l < r) {
        int mid = l + (r - l)/2;
        if (A[mid]==target){
            return mid;
        }
        if (A[mid]<target){
            l = mid + 1;
        }
        if (A[mid]>target){
            r = mid - 1;
        }
    }
    return l;
}

// longest valid brackets
int longestValidParentheses(string s) {
    stack<pair<int, int>> stk;
    int maxLen = 0;
    int curLen = 0;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] == '(') {
            stk.push(make_pair(i, 0));
        } else {
            if (stk.empty() || stk.top().second == 1) {
                stk.push(make_pair(i, 1));
            } else {
                stk.pop();
                if (stk.empty())
                    curLen = i + 1;
                else
                    curLen = i - stk.top().first;
                
                maxLen = max(maxLen, curLen);
            }
        }
    }
    
    return maxLen;
}

// substring with concatenation of all words
vector<int> findSubstring(string s, vector<string> L) {
    vector<int> res;
    int len = L[0].size();
    map<string, int> mp;
    for (int i = 0; i<L.size(); i++) {
        mp[L[i]]++;
    }
    
    int i = 0;
    while (i + L.size() * len - 1 < s.size()) {
        map<string, int>mp2;
        int j = 0;
        while (j < L.size()) {
            string sub = s.substr(i + j * len, len);
            if (mp.find(sub) == mp.end()) {
                break;
            } else {
                mp2[sub] ++;
                if (mp2[sub] > mp[sub]) {
                    break;
                }
                j++;
            }
        }
        if (j == L.size()) {
            res.push_back(i);
        }
        i++;
    }
    
    return res;
}

// Remove Element
int removeElement(int a[], int n, int element) {
    int i = 0;
    int j = 0;
    while (i < n) {
        if (a[i] != element) {
            a[j] = a[i];
            j++;
        }
        i++;
    }
    return j;
}

// generate brackets
// generate parantheses
void gp(string str, int l, int r) {
    if (l == 0 && r == 0) {
        cout << str << endl;
        return;
    }
    if (l > 0) {
        gp(str + "(", l-1, r+1);
    }
    if (r > 0) {
        gp(str + ")", l, r-1);
    }
}

bool isValid(string s) {
    // Start typing your C/C++ solution below
    // DO NOT write int main() function
    vector<char> sta;
    if(s.size() ==0) return false;
    sta.push_back(s[0]);
    for(int i =1; i< s.size(); i++)
    {
        if(s[i] == '(' || s[i] == '[' || s[i] == '{')
        {
            sta.push_back(s[i]);
            continue;
        }
        char current = sta.back();
        if(s[i] == ')' && current != '(')
            return false;
        if(s[i] == ']' && current != '[')
            return false;
        if(s[i] == '}' && current != '{')
            return false;
        sta.pop_back();
    }
    if(sta.size() !=0) return false;
    return true;
}

// letter combination of phone number
class phoneNumber {
public:
    string index[8];
    
    void solution(string digits, int curr, string c, vector<string> &res) {
        if (curr == digits.size()) {
            res.push_back(c);
            return;
        }
        int d = digits[curr] - '0' - 2;
        for (int i = 0; i < index[d].size(); i++) {
            c += index[d][i];
            solution(digits, curr + 1, c, res);
            c.resize(c.size() - 1);
        }
    }
    
    vector<string> letterComb(string digits) {
        index[0]="abc";
        index[1]="def";
        index[2]="ghi";
        index[3]="jkl";
        index[4]="mno";
        index[5]="pqrs";
        index[6]="tuv";
        index[7]="wxyz";
        vector<string> res;
        solution(digits, 0, "", res);
        return res;
    }
};

// two sum
// twosum
// 2sum
// 2 sum
bool twoSum(int a[], int n, int target) {
    map<int, int> mp;
    for (int i = 0; i < n; i++) {
        mp.insert(make_pair(a[i], i));
    }
    
    for (int i = 0; i < n; i++) {
        if (mp.find(target - a[i]) != mp.end()) {
            if (mp[target - a[i]] != i) {
                return true;
            }
        }
    }
    
    return false;
}

// three sum
vector<vector<int>> threeSum(vector<int> num) {
    sort(num.begin(), num.end());
    vector<vector<int>> res;
    for (int i = 0; i < num.size(); i++) {
        int start = i + 1;
        int end = num.size() - 1;
        while (start < end) {
            if (num[i] + num[start] + num[end]) {
                vector<int> oneRes(3);
                oneRes[0] = i;
                oneRes[1] = start;
                oneRes[2] = end;
                res.push_back(oneRes);
                start ++;
                end --;
            } else if (num[i] + num[start] + num[end] < 0)
                start ++;
            else
                end --;
            
        }
    }
    return res;
}

// three sum closest
int threeSumClosest(vector<int> num, int target) {
    int closest = num[0] + num[1] + num[2];
    int diff = abs(target - closest);
    sort(num.begin(), num.end());
    for (int i = 0; i < num.size() - 2; i++) {
        int start = i + 1;
        int end = num.size() - 1;
        while (start < end) {
            int sum = num[i] + num[start] + num[end];
            int newDiff = abs(sum - target);
            if (newDiff < diff) {
                diff = newDiff;
                closest = sum;
            }
            if (sum < target)
                start++;
            else
                end--;
        }
    }
    return closest;
}

// four sum
// foursum
vector<vector<int>> fourSum(vector<int> num, int target) {
    map<int, vector<pair<int, int>>> dict;
    
    vector<vector<int>> res;
    
    for (int i = 0; i<num.size(); i++) {
        for (int j = i+1; j < num.size(); j++) {
            int sum = num[i] + num[j];
            if (dict.find(sum) != dict.end()) {
                vector<pair<int, int>> sumPair;
                sumPair.push_back(make_pair(num[i], num[j]));
                dict.insert(make_pair(sum, sumPair));
            } else {
                dict[sum].push_back(make_pair(num[i], num[j]));
            }
        }
    }
    
    map<int, vector<pair<int, int>>>::iterator it;
    for (it = dict.begin(); it != dict.end(); it++) {
        vector<pair<int, int>> sumPair = it->second;
        if (dict.find(target-it->first) != dict.end()) {
            if (target - it->first == it->first && sumPair.size() == 1) {
                continue;
            }
            
            vector<pair<int, int>> secondPair = dict[target - it->first];
            
            for (auto pair1: sumPair) {
                for (auto pair2: secondPair) {
                    if (pair1 == pair2) {
                        continue;
                    }
                    
                    if (pair1.first == pair2.first || pair1.first == pair2.second) {
                        continue;
                    }
                    
                    vector<int> tmpRes(4);
                    tmpRes[0] = pair1.first;
                    tmpRes[1] = pair1.second;
                    tmpRes[3] = pair2.first;
                    tmpRes[4] = pair2.second;
                    res.push_back(tmpRes);
                }
            }
        }
        
    }
    
    return res;
}

int maxArea(vector<int> height) {
    int maxArea = 0;
    int l = 0;
    int r = height.size() - 1;
    
    while (l < r) {
        int area = abs(r - l) * min(height[l], height[r]);
        
        maxArea = max(maxArea, area);
        
        if (height[l] < height[r]) {
            l++;
        } else {
            r--;
        }
    }
    return maxArea;
}

// string to int
int atoi(const char *str) {
    // Start typing your C/C++ solution below
    // DO NOT write int main() function
    if (!str){return 0;}
    int i=0;
    bool pos=true;
    int res=0;
    while (str[i]==' '){ i++;}
    if (str[i]=='+'){ pos=true;i++;}
    if (str[i]=='-'){ pos = false;i++;}
    if (!isdigit(str[i])){return 0;}
    while (isdigit(str[i])){
        if (pos && res>INT_MAX/10){return INT_MAX;}
        if (pos && res==INT_MAX/10 && int(str[i]-'0')>=7){return INT_MAX;}
        if (!pos && -res<INT_MIN/10){return INT_MIN;}
        if (!pos && -res==INT_MIN/10 && int(str[i]-'0')>=8){return INT_MIN;}
        res = res*10 + int(str[i]-'0');
        i++;
    }
    
    if (pos){return res;}
    else{return -res;}
    
}

// reverse a int
// reverse number
int reverse (int x) {
    int res = 0;
    int left = 0;
    while (x != 0) {
        left = x % 10;
        res = res * 10 + left;
        x = x/10;
    }
    
    return res;
}

// length of longest with only unique
// length of longest without repeating
int lengthOfLongestSubstring(string s) {
    // Start typing your C/C++ solution below
    // DO NOT write int main() function
    if(s.size()==0){return 0;}
    if(s.size()==1){return 1;}
    int i=0;
    int j=0;
    int maxl = 0;
    bool table[256] = {false};
    while ( (i<s.size()) && (j<s.size()) ){
        if (table[s[j]]==false){
            table[s[j]]=true;
            maxl = max(maxl,j-i+1);
            j++;
        }else if (table[s[j]]==true){
            maxl = max(maxl,j-i);
            table[s[i]]=false;
            i++;
        }
    }
    return maxl;
}

// regular expression matching
// dynamic programming
bool isMatch_regularExpression(char *s, char *p) {
    int m = strlen(s);
    int n = strlen(p);
    vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
    dp[0][0] = true;
    
    for (int i = 0; i <=m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p[j-1] != '.' && p[j-1] != '*') {
                if (i > 0 && s[i-1] == p[j-1] && dp[i - 1][j - 1]) {
                    dp[i][j] = true;
                }
            } else if (p[j-1] == '.') {
                if (i > 0 && dp[i-1][j-1]) {
                    dp[i][j] = true;
                }
            } else if (j > 1 && p[j-1] == '*') { // dealing with the '*' case
                if (dp[i][j-1] || dp[i][j-2]) {
                    dp[i][j] = true;
                } else if (i>0 && (p[j-2] == s[i-1] || p[j-2] == '.') && dp[i-1][j]) {
                    dp[i][j] = true;
                }
            }
            
            
        }
    }
    return dp[m][n];
}

// recursion
bool isMatch_recursion(char *s, char *p) {
    assert(s && p);
    if (!*p) {
        return !*s;
    }
    
    if (*(p+1) != '*') {
        assert(*p != '*');
        return (*p == *s || (*p=='.' && *s)) && isMatch_recursion(s+1, p+1);
    }
    
    // next character is '*'
    while (*p == *s || (*p =='.' && *s)) {
        if (isMatch_recursion(s, p+2)) return true;
        s++;
    }
    
    return isMatch_recursion(s, p+2);
}

#define UNASSIGNED 0
#define N 9
// sudoku solver
bool findUnassignedLocation(int grid[N][N], int &row, int &col) {
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            if (grid[row][col] == UNASSIGNED) {
                return true;
            }
        }
    }
    return false;
}

bool usedInRow(int grid[N][N], int row, int num) {
    for (int col = 0; col < N; col ++) {
        if (grid[row][col] == num) {
            return true;
        }
    }
    return true;
}

bool usedInCol(int grid[N][N], int col, int num)
{
    for (int row = 0; row < N; row++)
        if (grid[row][col] == num)
            return true;
    return false;
}

bool usedInBox(int grid[N][N], int bosStartRow, int boxStartCol, int num) {
    for (int row = 0; row < 3; row ++ ) {
        for (int col = 0; col < 3; col++) {
            if (grid[row + bosStartRow][col + boxStartCol] == num) {
                return true;
            }
        }
    }
    return false;
}

bool isSafe(int grid[N][N], int row, int col, int num) {
    return !usedInBox(grid, row - row % 3, col - col % 3, num) && !usedInRow(grid, row, num) && !usedInCol(grid, col, num);
}

bool solveSudoku(int grid[N][N]) {
    int row, col;
    if (!findUnassignedLocation(grid, row, col)) {
        return true;
    }
    
    for (int i = 1; i <= 9; i++) {
        if (isSafe(grid, row, col, i)) {
            grid[row][col] = i;
            
            solveSudoku(grid);
            
            grid[row][col] = UNASSIGNED;
        }
    }
    return false;
}


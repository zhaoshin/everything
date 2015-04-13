//
//  sort.cpp
//  Everything
//
//  Created by Zhao, Xing on 3/19/15.
//  Copyright (c) 2015 Zhao, Xing. All rights reserved.
//

#include <stdio.h>


// number of occurence
// frequence of occurence
/* if x is present in arr[] then returns the index of FIRST occurrence
 of x in arr[0..n-1], otherwise returns -1 */
int first(int arr[], int low, int high, int x, int n)
{
    if(high >= low)
    {
        int mid = (low + high)/2;  /*low + (high - low)/2;*/
        if( ( mid == 0 || arr[mid] > arr[mid-1]) && arr[mid] == x)
            return mid;
        else if(x > arr[mid])
            return first(arr, (mid + 1), high, x, n);
        else
            return first(arr, low, (mid -1), x, n);
    }
    return -1;
}


/* if x is present in arr[] then returns the index of LAST occurrence
 of x in arr[0..n-1], otherwise returns -1 */
int last(int arr[], int low, int high, int x, int n)
{
    if(high >= low)
    {
        int mid = (low + high)/2;  /*low + (high - low)/2;*/
        if( ( mid == n-1 || arr[mid] < arr[mid+1]) && arr[mid] == x )
            return mid;
        else if(x < arr[mid])
            return last(arr, low, (mid -1), x, n);
        else
            return last(arr, (mid + 1), high, x, n);
    }
    return -1;
}

/* if x is present in arr[] then returns the count of occurrences of x,
 otherwise returns -1. */
int count(int arr[], int x, int n)
{
    int i; // index of first occurrence of x in arr[0..n-1]
    int j; // index of last occurrence of x in arr[0..n-1]
    
    /* get the index of first occurrence of x */
    i = first(arr, 0, n-1, x, n);
    
    /* If x doesn't exist in arr[] then return -1 */
    if(i == -1)
        return i;
    
    /* Else get the index of last occurrence of x. Note that we
     are only looking in the subarray after first occurrence */
    j = last(arr, i, n-1, x, n);
    
    /* return count */
    return j-i+1;
}

int rotated_binary_search(int A[], int N, int key) {
    int L = 0;
    int R = N - 1;
    
    while (L <= R) {
        // Avoid overflow, same as M=(L+R)/2
        int M = L + ((R - L) / 2);
        if (A[M] == key) return M;
        
        // the bottom half is sorted
        if (A[L] <= A[M]) {
            if (A[L] <= key && key < A[M])
                R = M - 1;
            else
                L = M + 1;
        }
        // the upper half is sorted
        else {
            if (A[M] < key && key <= A[R])
                L = M + 1;
            else
                R = M - 1;
        }
    }
    return -1;
}

// pivot
void swap(int& a, int& b)
{
    a -= b;
    b += a;// b gets the original value of a
    a = (b - a);// a gets the original value of b
}

int pivot(int a[], int first, int last) {
    int p = first;
    int pivotElement = a[first];
    
    for (int i = first + 1; i <= last; i++) {
        if (a[i] <= pivotElement) {
            p++;
            swap(a[i], a[p]);
        }
    }
    
    swap(a[p], a[first]);
    
    return p;
}

// select the kth number
int quick_select(int a[], int l, int r, int k) {
    if (l == r) {
        return a[l];
    }
    
    int j = pivot(a, l, r);
    
    int length = j - l + 1;
    
    if (length == k) {
        return a[j];
    } else if (k < length) {
        return quick_select(a, l, j - 1, k);
    } else {
        return quick_select(a, j + 1, r, k - length);
    }
}

#define RANGE 255

void countSort(int *a, int size) {
    int output[size];
    
    int count[RANGE + 1] = {0};
    
    for (int i = 0; i < size; i++) {
        count[a[i]]++;
    }
    
    for (int i = 1; i <= RANGE; i++)
        count[i] += count[i-1];
    
    for (int i = 0; i < size; i++) {
        output[count[a[i]] - 1] = a[i];
        count[a[i]]--;
    }
    
    for (int i = 0; i < size; i++) {
        a[i] = output[i];
    }
    
}

void quickSort( int a[], int first, int last )
{
    int pivotElement;
    
    if(first < last)
    {
        pivotElement = pivot(a, first, last);
        quickSort(a, first, pivotElement-1);
        quickSort(a, pivotElement+1, last);
    }
}

void merge(int array[], int low, int mid, int high, int &res) {
    int i, j, k, c[100];
    i = low;
    k = low;
    j = mid + 1;
    while (i <= mid && j <= high) {
        
        if (array[i]<array[j]) {
            c[k] = array[i];
            k++;
            i++;
        }
        else {
            res += mid - low + 1;
            c[k] = array[j];
            k++;
            j++;
        }
    }
    while (i <= mid) {
        c[k] = array[i];
        k++;
        i++;
    }
    while (j <= high) {
        c[k] = array[j];
        k++;
        j++;
    }
    
    for (i = low; i <= high; i++) {
        array[i] = c[i];
    }
}

void my_merge_sort(int array[], int low, int high, int &res) {
    int mid;
    if (low < high) {
        
        mid = (low + high) / 2;
        my_merge_sort(array, low, mid, res);
        my_merge_sort(array, mid+1, high, res);
        merge(array, low, mid, high, res);
    }
}

// Heap sort
// heapsort

// maintain heap relationship for the triangle from low to high
void shiftRight(int *arr, int low, int high) {
    int root = low;
    while (root * 2 + 1 <= high) {
        int leftChild = root * 2 + 1;
        int rightChild = leftChild + 1;
        int swapIndex = root;
        
        if (arr[swapIndex] < arr[leftChild]) {
            swapIndex = leftChild;
        }
        
        if (rightChild <= high && arr[swapIndex] < arr[rightChild]) {
            swapIndex = rightChild;
        }
        
        if (swapIndex != root) {
            swap(arr[root], arr[swapIndex]);
            
            root = swapIndex;
        } else {
            break;
        }
    }
}

// create heap structure
// call shiftRigth from last triangle to the first triangle
void heapify(int *arr, int low, int high) {
    int mid = low + (high - low)/2;
    while (mid >= 0) {
        shiftRight(arr, mid, high);
        
        mid --;
    }
}

// heapsort
void heapSort(int *arr, int size) {
    heapify(arr, 0, size-1);
    
    int high = size - 1;
    while (high > 0) {
        swap(arr[high], arr[0]);
        high -- ;
        shiftRight(arr, 0, high);
    }
    
}
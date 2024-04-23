#include<iostream>
#include<omp.h>
#include<bits/stdc++.h>
#include<chrono>  // Include chrono header for duration calculation
 
using namespace std;

void minimumSeq(vector<int> array){
    int min = INT_MAX; 
    for(auto i = array.begin(); i != array.end();i++){
        if(*i < min){
            min = *i;
        } 
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Minimum Element: " << min << endl;

}

void minimumPara(vector<int> array){
    int min_ele = INT_MAX;
    #pragma omp parallel for reduction(min: min_ele)
        for(auto i = array.begin(); i != array.end();i++){
            if(*i < min_ele){
                min_ele = *i;
            } 
        }
}

void maximumSeq(vector<int> array){
    int max = INT_MIN;
    
    for(auto i = array.begin(); i != array.end();i++){
        if(*i > max){
            max = *i;
        } 
    }
    
    cout << "Maximum Element: " << max << endl;

}

void maximumPara(vector<int> array){
    int max_ele = INT_MIN;
    
    #pragma omp parallel for reduction(max: max_ele)
        for(auto i = array.begin(); i != array.end();i++){
            if(*i > max_ele){
                max_ele = *i;
            } 
        }
        
    cout << "Maximum Element(Parallel Reduction): " << max_ele << endl;

}

void sumSeq(vector<int> array){
    int sum = 0;
    
    for(auto i = array.begin(); i != array.end();i++){
        sum += *i; 
    }
    
    cout << "Summation: " << sum << endl;
}

void sumPara(vector<int> array){
    int sum = 0;
    
    #pragma omp parallel for reduction(+: sum)
        for(auto i = array.begin(); i != array.end();i++){
            sum += *i; 
        }
        
    cout << "Summation(Parallel Reduction): " << sum << endl;
    
}

void averageSeq(vector<int> array){
    float avg = 0;
    
    for(auto i = array.begin(); i != array.end();i++){
        avg += *i; 
    }
    
    cout << "Average: " << avg / array.size() << endl;

}

void averagePara(vector<int> array){
    int avg = 0;
    
    #pragma omp parallel for reduction(+: avg)
        for(auto i = array.begin(); i != array.end();i++){
            avg += *i; 
        }
        
    cout << "Average(Parallel Reduction): " << avg / array.size() << endl;
}

int main(){
    cout << "Enter number of elements in array: ";
    int N;
    int MAX = 10;
    cin >> N;
    vector<int> array;
    for(int i = 0; i < N; i++){
        array.push_back(rand() % MAX);
    }

    cout<<"Elements of the array : "<<endl;

    for(auto i: array){
        cout<< i << " ";
    }
    cout<<endl;

    // Sequential Min
    auto start = chrono::high_resolution_clock::now();
    minimumSeq(array);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> dur = end - start;
    cout << "Sequential Time for Minimum: " << dur.count() << endl;

    cout<<typeid(start).name()<<endl;

    // Parallel Min
    start = chrono::high_resolution_clock::now();
    minimumPara(array);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> durPara = end - start;
    cout << "Parallel Time for Minimum: " << durPara.count() << endl;

    
    // Sequential Maximum
    start = chrono::high_resolution_clock::now();
    maximumSeq(array);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dur_seq_max = end - start;
    cout << "Sequential Time for Maximum: " << dur_seq_max.count() << endl;

    // Parallel Maximum
    start = chrono::high_resolution_clock::now();
    maximumPara(array);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dur_para_max = end - start;
    cout << "Parallel Time for Maximum: " << dur_para_max.count() << endl;

    
    // Sequential Average
    start = chrono::high_resolution_clock::now();
    averageSeq(array);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dur_seq_avg = end - start;
    cout << "Sequential Time for Average: " << dur_seq_avg.count() << endl;

    // Parallel Average
    start = chrono::high_resolution_clock::now();
    averagePara(array);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dur_para_avg = end - start;
    cout << "Parallel Time for Average: " << dur_para_avg.count() << endl;
    

    
    // Sequential Sum
    start = chrono::high_resolution_clock::now();
    sumSeq(array);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dur_seq_sum = end - start;
    cout << "Sequential Time for Sum: " << dur_seq_sum.count() << endl;

    // Parallel Sum
    start = chrono::high_resolution_clock::now();
    sumPara(array);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dur_para_sum = end - start;
    cout << "Parallel Time for Sum: " << dur_para_sum.count() << endl;

        
    // Calculate speedup for Minimum
    double speedup = dur.count() / durPara.count();
    cout << "Speedup (Minimum): " << speedup*10 << endl;
        
    // Calculate speedup for Maximum
    double speedup_max = dur_seq_max.count() / dur_para_max.count();
    cout << "Speedup (Maximum): " << speedup_max*10 << endl;

    // Calculate speedup for Average
    double speedup_avg = dur_seq_avg.count() / dur_para_avg.count();
    cout << "Speedup (Average): " << speedup_avg*10 << endl;

    // Calculate speedup for Sum
    double speedup_sum = dur_seq_sum.count() / dur_para_sum.count();
    cout << "Speedup (Sum): " << speedup_sum*10 << endl;
    
    return 0;
}

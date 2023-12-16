#include <stdio.h>
#include <math.h>
#include <stdlib.h>
int main(){

int *input, *filter,*feature_map;
int n=4;
int m=3;
int cnt=0;
int cnt_c=0;
int feature[9] = {0};
int input[16]= {1,0,0,0,1,1,1,1,0,1,0,0,1,1,0,1};
int filter[9] = {1,0,0,0,1,1,1,1,1};


printf("matrix A : \n");
for(int x=0; x<(n-m+1)*(n-m+1); x++){
    for(int i = 0; i < m; ++i){        
        for(int j = 0; j < m; ++j){ //m*n ma
            feature[i*n+j] = input[cnt]*filter[i*n+j];
            printf("%d\t ", feature[i*n +j]);
            cnt += 1;
        }
        cnt += 1;
        printf("\n");
    }
    if(m-n-x == 0) {    cnt_c+=m;   }
    else           {   cnt_c+=1;   }
    printf("\n");
    feature[9] = {0};
}



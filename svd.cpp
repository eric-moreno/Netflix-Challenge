#include <iostream>
#include <fstream>
#include <string>
#include </Users/Johanna/Desktop/Eigen/Eigen/SVD>
#include </Users/Johanna/Desktop/Eigen/Eigen/Dense>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;

int NUM_USERS = 458293;
int NUM_MOVIES = 17770;

int main()
{
   MatrixXf A;
   // Read training data from "base.txt"
   ifstream base;
   base.open("base.txt");
   while (! base.eof()) {
      int i = 0;
      float u, m, t, r;
      while(base >> u >> m >> t >> r) {
          A.row(i) << u, m, t, r;
          i++;
      }
   }
    
    //WTF is this matrix???
   MatrixXf A = MatrixXf::Random(100, 4);
   BDCSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);
   std::cout << "..." << std::endl;

   const Eigen::MatrixXf U = svd.matrixU();
   const Eigen::MatrixXf V = svd.matrixV();  

   // Predict
   ifstream qual_file ("qual_toy.txt");
   //float predictions [26];
   
   if (qual_file.is_open()) {
       float i, j;
       while (qual_file >> i >> j) {
           float prediction = (U.transpose())(i-1) * V(j-1);
           printf("prediction: %.3f\n", prediction);
       }
       qual_file.close();
   }
   
   
   return 0;
}
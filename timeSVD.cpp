// timeSVD.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;

const int userCount = 458293;
const int movieCount = 17770;
const double avgRating = 3.60073; 
const int k = 40;
double* Bu; 
double* Bi;
double** Pu;
double** Qi; 

double** R; 

int main()
{

	// Initialize R 
	R = new double*[userCount];

	for (int i = 0; i < userCount; i++)
	{
		R[i] = new double[movieCount];
	}

	for (int i = 0; i < userCount; i++)
	{
		for (int j = 0; j < movieCount; j++)
		{
			R[i][j] = -1.0; 
		}
	}

	// Read input file all.dta
	ifstream file("E:\\Documents\\Caltech\\CS156\\mu\\all.dta");
	string input;
	int user, movie, time, rating;
	int index = 0; 
	while(getline(file, input))
	{
		sscanf_s(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);
		R[user - 1][movie - 1] = rating; 

		//train[user - 1].push_back(make_pair(make_pair(movie, rating), time));
	}

	file.close();

	// Now we deal with no ratings (-1 values) 
	double rowAverage[userCount];
	for (int i = 0; i < userCount; i++)
	{
		int sum = 0;
		for (int j = 0; j < movieCount; j++)
		{
			if (R[i][j] >= 0)
			{
				sum += R[i][j];
			}
		}
		rowAverage[i] = sum; 
	}

	double colAverage[movieCount];
	for (int i = 0; i < movieCount; i++)
	{
		int sum = 0;
		for (int j = 0; j < userCount; j++)
		{
			if (R[j][i] >= 0)
			{
				sum += R[j][i];
			}
		}
		colAverage[i] = sum;
	}

	// Set all missing values to column average
	for (int i = 0; i < movieCount; i++)
	{
		double col = colAverage[i];
		for (int j = 0; j < userCount; j++)
		{
			if (R[j][i] < 0)
			{
				R[j][i] = col; 
			}
		}
	}

	// Subtract row average from all values 
	for (int i = 0; i < userCount; i++)
	{
		double sub = rowAverage[i];
		for (int j = 0; j < movieCount; j++)
		{
			R[i][j] = R[i][j] - sub;
		}
	}

	// Sketchy 
	Eigen::MatrixXd RMatrix(userCount, movieCount);
	for (int i = 0; i < userCount; i++)
		RMatrix.row(i) = Eigen::VectorXd::Map(&R[i][0], movieCount);

	BDCSVD<MatrixXd> svd(RMatrix, ComputeFullV | ComputeFullU);

	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::MatrixXd S = svd.singularValues();

	// Dimensionality reduction
	U.resize(userCount, k);
	V.resize(k, movieCount);
	S.resize(k, k); 

	Eigen::MatrixXd sqrtS = S.sqrt();
	Eigen::MatrixXd UserK = U * sqrtS.transpose();
	Eigen::MatrixXd MovieK = sqrtS * V.transpose();

	ifstream qualfile("E:\\Documents\\Caltech\\CS156\\mu\\qual.dta");
	int user, movie, time, rating;
	int index = 0;
	while (getline(qualfile, input))
	{
		sscanf_s(input.c_str(), "%d %d %d %d", &user, &movie, &time);
		R[user - 1][movie - 1] = rating;

		//train[user - 1].push_back(make_pair(make_pair(movie, rating), time));
	}

	qualfile.close();

	/*
	//Initialize Bu
	Bu = new double[userCount];
	for (size_t i = 0; i < userCount; i++) {
		Bu[i] = 0.0;
	}

	//Initialize Bi
	Bi = new double[movieCount];
	for (size_t i = 0; i < movieCount; i++) {
		Bi[i] = 0.0;
	}

	//Initialize Pu
	Pu = new double*[userCount];
	for (size_t i = 0; i < userCount; i++) {
		Pu[i] = new double[factors];
	}

	for (size_t i = 0; i < userCount; i++) 
	{
		for (size_t j = 0; j < factors; j++)
		{
			Pu[i][j] = rand() / RAND_MAX / sqrt(factors);
		}
	}

	*/
	//Initialize Qi



	std::cout << "hello";
	
	std::cin.get();
    return 0;
}

